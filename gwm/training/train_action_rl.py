

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwm.training.runtime_cpu import configure_cpu_environment, configure_torch_runtime

_CPU_THREADS = configure_cpu_environment(overwrite=True)

import torch
import numpy as np
from torch.func import vmap
from torch.nn import functional as F

configure_torch_runtime(torch, _CPU_THREADS)

from gwm.action_rl import (
    ActionAwareController,
    DynamicGroupSampler,
    RewardedSequenceRollout,
    calibrated_soft_node_action_probability,
    edge_representation_change_reward,
    grpo_sequence_loss,
    node_change_localization_reward,
    node_property_reward,
    node_state_magnitude_reward,
    structure_node_weights,
    structure_weighted_f1,
    topology_evolution_reward,
)
from gwm.actions import (
    GraphActionEncoder,
    GraphEditSequence,
    GraphOperation,
    apply_graph_edit_sequence,
    bounded_graph_edit_sequence,
)
from gwm.data.action_benchmark import (
    ACTION_BENCHMARK_DATASETS,
    ActionBenchmarkTransitionDataset,
    infer_action_benchmark_dataset,
    resolve_action_benchmark_path,
)
from gwm.latent_objective import latent_transition_terms
from gwm.model import GraphWorldModel
from gwm.transfer import FrozenGraphTransferBranch, GatedResidualTransferAdapter
from gwm.pretraining import (
    load_worldgraph_pretrained_backbone,
    load_worldgraph_pretrained_task_module,
)
from gwm.pair_state import CausalPairState, PAIR_STATE_DIM
from gwm.genre_history_state import GenreObservedPropertyHistory
from gwm.property_transition import decode_property_prediction
from gwm.utils import count_parameters, resolve_device, save_json, seed_everything


TASK_OPERATIONS = {
    "topology": (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE),
    "node_change": (GraphOperation.MODIFY_NODE_STATE,),
    "node_state": (GraphOperation.MODIFY_NODE_STATE,),
    "node_property": (GraphOperation.MODIFY_NODE_PROPERTY,),
}


def _action_groups_for_step(
    task: str,
    *,
    joint_action_groups: bool,
    global_step: int,
    interval: int,
) -> tuple[tuple[GraphOperation, ...], bool]:
    active = bool(
        joint_action_groups
        and global_step % interval == 0
    )
    native = TASK_OPERATIONS[task]
    if not active:
        return native, False
    if task == "topology":
        return (*native, GraphOperation.MODIFY_NODE_STATE), True
    return (GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE, *native), True


@contextmanager
def _preserve_world_rng(device: torch.device):

    devices: list[int] = []
    if device.type == "cuda":
        devices = [
            torch.cuda.current_device()
            if device.index is None
            else int(device.index)
        ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


@contextmanager
def _preserve_validation_process_state(device: torch.device):

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    grad_enabled = torch.is_grad_enabled()
    default_dtype = torch.get_default_dtype()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    current_cuda_device = (
        torch.cuda.current_device()
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    try:
        with _preserve_world_rng(device):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_grad_enabled(grad_enabled)
        torch.set_default_dtype(default_dtype)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if current_cuda_device is not None:
            torch.cuda.set_device(current_cuda_device)


class _PersistentValidationWorker:

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "gwm" / "training" / "persistent_action_validation_worker.py"),
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        ready_line = self.process.stdout.readline()
        if not ready_line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "Persistent validation worker failed during startup: " + stderr
            )
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(
                f"Unexpected persistent validation worker response: {ready!r}"
            )

    def evaluate(self, arguments: list[str]) -> str:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Persistent validation worker exited with {self.process.returncode}."
            )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps({"args": arguments}) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "Persistent validation worker stopped without a response: " + stderr
            )
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(
                "Persistent validation failed:\n"
                + str(response.get("traceback", response.get("stderr", "")))
            )
        return str(response.get("stderr", ""))

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _encode_action_or_zero(
    encoder: GraphActionEncoder,
    z_t: torch.Tensor,
    action: GraphEditSequence,
    *,
    enabled: bool,
) -> torch.Tensor:
    if enabled:
        return encoder(z_t, action)["nodewise"]
    return z_t.new_zeros((z_t.shape[0], encoder.action_dim))


def _action_transition_advantage_loss(
    action_latent_mu: torch.Tensor,
    zero_action_latent_mu: torch.Tensor,
    target_z: torch.Tensor,
    *,
    margin: float,
) -> dict[str, torch.Tensor]:
    if margin < 0.0:
        raise ValueError("Action-transition advantage margin must be non-negative.")
    if (
        action_latent_mu.shape != zero_action_latent_mu.shape
        or action_latent_mu.shape != target_z.shape
    ):
        raise ValueError(
            "Action, zero-action, and target latent tensors must have identical shapes."
        )
    action_mse = F.mse_loss(action_latent_mu, target_z)
    zero_action_mse = F.mse_loss(
        zero_action_latent_mu.detach(), target_z.detach()
    )
    advantage = F.relu(action_mse - zero_action_mse + float(margin))
    return {
        "total": advantage,
        "action_mse": action_mse,
        "zero_action_mse": zero_action_mse,
    }


def _action_observable_advantage_loss(
    task: str,
    action_terms: dict[str, torch.Tensor],
    zero_action_terms: dict[str, torch.Tensor],
    *,
    margin: float,
) -> dict[str, torch.Tensor]:
    if margin < 0.0:
        raise ValueError("Action-observable advantage margin must be non-negative.")
    if task == "topology":
        key = "observable"
    elif task == "node_change":


        key = "change"
    elif task in {"node_state", "node_property"}:
        key = "observable"
    else:
        raise ValueError(f"Unsupported task for observable advantage: {task!r}")
    action_value = action_terms[key]
    zero_value = zero_action_terms[key].detach()
    advantage = F.relu(action_value - zero_value + float(margin))
    return {
        "total": advantage,
        "action_observable": action_value,
        "zero_action_observable": zero_value,
    }


def _batched_action_model_outputs(
    model: GraphWorldModel,
    x_t: torch.Tensor,
    edge_index_t: torch.Tensor,
    hidden: torch.Tensor,
    action_batch: torch.Tensor,
    *,
    edge_weight_t: torch.Tensor | None,
    addition_candidate_edges: torch.Tensor | None,
    deletion_candidate_edges: torch.Tensor | None,
    addition_pair_features: torch.Tensor | None,
    deletion_pair_features: torch.Tensor | None,
    decode_node_state: bool,
    precomputed_z_t_raw: torch.Tensor,
) -> dict[str, torch.Tensor]:

    def one(action: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = model(
            x_t,
            edge_index_t,
            hidden,
            action=action,
            edge_weight_t=edge_weight_t,
            addition_candidate_edges=addition_candidate_edges,
            deletion_candidate_edges=deletion_candidate_edges,
            addition_pair_features=addition_pair_features,
            deletion_pair_features=deletion_pair_features,
            decode_node_state=decode_node_state,
            commit_history=False,
            precomputed_z_t_raw=precomputed_z_t_raw,
        )
        return {key: value for key, value in outputs.items() if torch.is_tensor(value)}

    return vmap(one, randomness="different")(action_batch)


def _validation_selection_score(
    task: str,
    payload: dict[str, Any],
    *,
    node_state_metric: str = "rmse",
    node_change_metric: str = "auprc",
) -> float:
    formal = payload["formal_metrics"]
    if task == "topology":





        return (
            float(formal["edge_addition"]["f1"])
            + float(formal["edge_deletion"]["f1"])
        ) / 2.0
    if task == "node_change":
        calibrated = formal["validation_calibrated"]
        auprc = calibrated["auprc"]
        f1 = calibrated["f1"]
        if node_change_metric == "auprc":
            value = auprc
        elif node_change_metric == "f1":
            value = f1
        else:





            if auprc is None or f1 is None:
                value = None
            else:
                denominator = float(auprc) + float(f1)
                value = (
                    0.0
                    if denominator <= 0.0
                    else 2.0 * float(auprc) * float(f1) / denominator
                )
        return float("-inf") if value is None else float(value)
    if task == "node_state":
        return -float(formal["changed_nodes"][node_state_metric])




    return float(formal["val"]["gwm"]["official_ndcg_at_10"])


def _edge_weight(snapshot: dict[str, Any], edge_input: str) -> torch.Tensor | None:
    if edge_input == "binary":
        return None
    weight = snapshot.get("edge_weight_t", snapshot.get("edge_weight"))
    if weight is None:
        raise ValueError("edge_input=weighted requires observed edge weights.")
    return weight


def _transition_to_device(
    transition: dict[str, Any],
    device: torch.device,
    *,
    task: str,
) -> dict[str, Any]:

    if task != "node_property":
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in transition.items()
        }
    tensor_keys = {
        "x_t",
        "edge_index_t",
        "edge_weight_t",
        "property_node_ids",
        "property_target",
        "property_current_target",
        "property_current_observed_mask",
        "property_observation_mask_t",
    }
    moved = {
        key: (
            value.to(device)
            if torch.is_tensor(value) and key in tensor_keys
            else value
        )
        for key, value in transition.items()
        if not torch.is_tensor(value) or key in tensor_keys
    }
    property_slice = moved["property_slice"]
    moved["property_t"] = moved["x_t"][:, property_slice]
    return moved


def _warm_start_clip_hidden_from_observed_prefix(
    model: GraphWorldModel,
    transitions: Any,
    hidden: torch.Tensor,
    *,
    transition_index: int,
    burn_in: int,
    device: torch.device,
    task: str,
    edge_input: str,
    property_history_input: bool = False,
) -> torch.Tensor:

    if burn_in <= 0 or transition_index <= 0:
        return hidden
    if task == "node_property" and property_history_input:
        return hidden

    begin = max(0, int(transition_index) - int(burn_in))
    was_training = model.training


    with _preserve_world_rng(device), torch.no_grad():
        model.eval()
        for prefix_index in range(begin, int(transition_index)):
            prefix = _transition_to_device(
                transitions[prefix_index], device, task=task
            )
            edge_weight = _edge_weight(prefix, edge_input)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            outputs = model(
                prefix["x_t"],
                prefix["edge_index_t"],
                hidden,
                action=None,
                edge_weight_t=edge_weight,
                topology_cache_key=int(prefix.get("transition_id", prefix_index)),
                decode_node_state=False,
                commit_history=True,
            )
            hidden = outputs["hidden_next"].detach()
    model.train(was_training)
    return hidden


def _warm_start_unary_action_history_from_observed_prefix(
    model: GraphWorldModel,
    action_encoder: GraphActionEncoder,
    controller: ActionAwareController,
    transitions: Any,
    hidden: torch.Tensor,
    *,
    transition_index: int,
    burn_in: int,
    device: torch.device,
    task: str,
    edge_input: str,
    soft_node_action_mode: str,
    soft_node_action_topk: int,
    soft_node_action_probability_calibration: str,
    soft_node_action_prior_smoothing: float,
    confidence_gated_soft_action: bool,
    soft_node_magnitude_conditioning: bool,
    change_gated_action_value: bool = False,
) -> torch.Tensor:

    if task not in {"node_change", "node_state"}:
        raise ValueError("Unary action-history replay supports only T2/T3 tasks.")
    if burn_in <= 0 or transition_index <= 0:
        return hidden

    begin = max(0, int(transition_index) - int(burn_in))
    num_nodes = int(hidden.shape[0])



    past_change_counts = torch.zeros(num_nodes, device=device)
    past_change_observations = torch.zeros((), device=device)
    for prefix_index in range(begin):
        prefix_cpu = transitions[prefix_index]
        past_change_counts += _past_change_update(
            task, prefix_cpu, device, num_nodes=num_nodes
        )
        past_change_observations += _past_change_observation_count(
            task, prefix_cpu, device
        )

    model_was_training = model.training
    action_encoder_was_training = action_encoder.training
    controller_was_training = controller.training
    with _preserve_world_rng(device), torch.no_grad():
        model.eval()
        action_encoder.eval()
        controller.eval()
        for prefix_index in range(begin, int(transition_index)):
            prefix_cpu = transitions[prefix_index]
            prefix = _transition_to_device(prefix_cpu, device, task=task)
            edge_weight = _edge_weight(prefix, edge_input)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            encode_observed_graph = getattr(model, "encode_observed_graph", None)
            if callable(encode_observed_graph):
                z_policy_raw = encode_observed_graph(
                    prefix["x_t"],
                    prefix["edge_index_t"],
                    edge_weight,
                    topology_cache_key=int(
                        prefix.get("transition_id", prefix_index)
                    ),
                )
            else:


                z_policy_raw = model.graph_encoder(
                    prefix["x_t"], prefix["edge_index_t"], edge_weight
                )
            z_policy = model._normalize_latent(z_policy_raw)
            controller_state = controller.node_state(z_policy, hidden)
            probability = calibrated_soft_node_action_probability(
                controller.node_target_head(controller_state).squeeze(-1),
                historical_positive_count=past_change_counts.sum(),
                historical_observation_count=past_change_observations,
                mode=soft_node_action_probability_calibration,
                smoothing=soft_node_action_prior_smoothing,
            )
            value = None
            if task == "node_state" and soft_node_magnitude_conditioning:
                value = controller._magnitude_distribution(
                    controller_state, GraphOperation.MODIFY_NODE_STATE
                )[0]
            encoded_action = action_encoder.encode_soft_node_action(
                z_policy,
                probability,
                GraphOperation.MODIFY_NODE_STATE,
                node_value=value,
                change_gated_value=change_gated_action_value,
                confidence_gate=confidence_gated_soft_action,
                localization_mode=soft_node_action_mode,
                top_k=soft_node_action_topk,
                support_count=None,
            )["nodewise"]
            outputs = model(
                prefix["x_t"],
                prefix["edge_index_t"],
                hidden,
                action=encoded_action,
                edge_weight_t=edge_weight,
                decode_node_state=False,
                commit_history=True,
                precomputed_z_t_raw=z_policy_raw,
            )
            hidden = outputs["hidden_next"].detach()
            past_change_counts += _past_change_update(
                task, prefix_cpu, device, num_nodes=num_nodes
            )
            past_change_observations += _past_change_observation_count(
                task, prefix_cpu, device
            )
    model.train(model_was_training)
    action_encoder.train(action_encoder_was_training)
    controller.train(controller_was_training)
    return hidden


def _warm_start_property_clip_hidden_from_observed_prefix(
    model: GraphWorldModel,
    transitions: Any,
    hidden: torch.Tensor,
    property_history: GenreObservedPropertyHistory,
    *,
    transition_index: int,
    burn_in: int,
    device: torch.device,
    edge_input: str,
) -> torch.Tensor:

    if burn_in <= 0 or transition_index <= 0:
        return hidden
    begin = max(0, int(transition_index) - int(burn_in))
    history = property_history.copy()
    was_training = model.training
    with _preserve_world_rng(device), torch.no_grad():
        model.eval()
        for prefix_index in range(begin, int(transition_index)):
            prefix_cpu = transitions[prefix_index]
            history.observe(
                prefix_cpu["property_node_ids_t"],
                prefix_cpu["property_observed_t"],
            )
            prefix = _transition_to_device(
                prefix_cpu, device, task="node_property"
            )
            prefix["x_t"] = torch.cat(
                [prefix["x_t"], history.state().to(device)], dim=-1
            )
            edge_weight = _edge_weight(prefix, edge_input)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            outputs = model(
                prefix["x_t"],
                prefix["edge_index_t"],
                hidden,
                action=None,
                edge_weight_t=edge_weight,
                decode_node_state=False,
                commit_history=True,
            )
            hidden = outputs["hidden_next"].detach()
    model.train(was_training)
    return hidden


def _scatter_property_state(
    snapshot: dict[str, Any], *, num_nodes: int, property_dim: int
) -> torch.Tensor:
    if "property_state" in snapshot:
        return snapshot["property_state"]
    state = torch.zeros((num_nodes, property_dim), dtype=torch.float32)
    state[snapshot["property_node_ids"].long()] = snapshot["property_target"].float()
    return state


def _graph_transitions(
    source: Path,
    task: str,
    split: str | None = "train",
    *,
    dataset: str | None = None,
    edge_input: str = "binary",
) -> tuple[dict[str, Any], ActionBenchmarkTransitionDataset]:
    transitions = ActionBenchmarkTransitionDataset(
        source,
        task=task,
        dataset=dataset,
        split=split,
        edge_input=edge_input,
        include_property_observation_mask=True,
    )
    return transitions.metadata, transitions


def _train_property_change_threshold(
    transitions: ActionBenchmarkTransitionDataset,
    *,
    quantile: float = 0.75,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    magnitudes: list[torch.Tensor] = []
    for index in range(len(transitions)):
        transition = transitions[index]
        if str(transition.get("split", "train")) != "train":
            continue
        comparable = transition["property_current_observed_mask"].to(torch.bool)
        if not bool(comparable.any()):
            continue
        delta = (
            transition["property_target"].float()
            - transition["property_current_target"].float()
        )
        magnitudes.append(delta[comparable].abs().mean(dim=-1).cpu())
    if not magnitudes:
        raise ValueError(
            "Cannot fit property-change threshold without comparable train rows."
        )
    return float(torch.quantile(torch.cat(magnitudes), float(quantile)).item())


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.to(torch.long)
    codes = torch.unique(edge_index[0].long() * num_nodes + edge_index[1].long())
    return torch.stack([codes // num_nodes, codes % num_nodes])


def _topology_count_context(
    edge_index_t: torch.Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:



    current_edges = int(edge_index_t.shape[1])
    possible = max(int(num_nodes) * max(int(num_nodes) - 1, 1), 1)
    values = [
        float(torch.log1p(torch.tensor(float(current_edges))).item()),
        float(torch.log1p(torch.tensor(float(num_nodes))).item()),
        float(current_edges / possible),
        float(torch.log1p(torch.tensor(float(current_edges) / max(num_nodes, 1))).item()),
    ]
    return torch.tensor(
        values,
        device=edge_index_t.device,
        dtype=dtype or torch.float32,
    )


def _topology_candidates(
    edge_index_t: torch.Tensor,
    *,
    num_nodes: int,
    metadata: dict[str, Any],
    max_addition_candidates: int,
    generator: torch.Generator,
    sample_without_materializing: bool = False,
    candidate_policy: str = "all_pairs",
    seen_codes: torch.Tensor | None = None,
) -> dict[GraphOperation, torch.Tensor]:
    if candidate_policy not in {"all_pairs", "seen_pairs"}:
        raise ValueError("candidate_policy must be all_pairs or seen_pairs")
    current = _unique_edges(edge_index_t.cpu(), num_nodes)
    current_codes = current[0] * num_nodes + current[1]




    coordinate_count = metadata.get("num_genre_coordinate_nodes")
    if coordinate_count is None:
        coordinate_count = metadata.get("num_target_coordinate_nodes")
    bipartite = coordinate_count is not None
    if bipartite:
        destination_start = 0
        destination_count = int(coordinate_count)
        source_start = destination_count
        source_count = num_nodes - destination_count
    elif metadata.get("tgb_src_dst_id_intersection") == 0 and metadata.get(
        "tgb_src_unique_node_ids"
    ):
        bipartite = True
        source_start = 0
        source_count = int(metadata["tgb_src_unique_node_ids"])
        destination_start = source_count
        destination_count = num_nodes - source_count
    else:
        source_start = destination_start = 0
        source_count = destination_count = num_nodes

    dataset_name = str(metadata.get("dataset_name", "")).lower()
    undirected = False
    include_self = bool(
        not bipartite
        and not undirected
        and (
            "including observed self"
            in str(metadata.get("candidate_space_definition", "")).lower()
            or "tgbn-trade" in dataset_name
        )
    )
    rectangular_pair_count = source_count * destination_count
    legal_pair_count = rectangular_pair_count
    if undirected:
        legal_pair_count = num_nodes * (num_nodes - 1) // 2
    elif not bipartite and not include_self:
        legal_pair_count -= num_nodes

    current_legal_codes: set[int] = set()
    for source, destination in current.t().tolist():
        if undirected and source > destination:
            source, destination = destination, source
        source_in_range = source_start <= source < source_start + source_count
        destination_in_range = (
            destination_start
            <= destination
            < destination_start + destination_count
        )
        if (
            source_in_range
            and destination_in_range
            and (include_self or source != destination)
        ):
            current_legal_codes.add(source * num_nodes + destination)
    available_count = max(0, legal_pair_count - len(current_legal_codes))






    if candidate_policy == "seen_pairs":
        historical = (
            torch.unique(seen_codes.detach().cpu().long())
            if seen_codes is not None and seen_codes.numel()
            else torch.empty(0, dtype=torch.long)
        )
        if historical.numel():
            h_source = historical // num_nodes
            h_destination = historical % num_nodes
            legal = (
                (h_source >= source_start)
                & (h_source < source_start + source_count)
                & (h_destination >= destination_start)
                & (h_destination < destination_start + destination_count)
            )
            if not include_self and not undirected:
                legal &= h_source.ne(h_destination)
            if undirected:
                legal &= h_source.lt(h_destination)
            historical = historical[legal]
            if historical.numel():
                historical = historical[~torch.isin(historical, current_codes)]
        additions = (
            torch.stack([historical // num_nodes, historical % num_nodes])
            if historical.numel()
            else torch.empty((2, 0), dtype=torch.long)
        )
        if max_addition_candidates > 0 and additions.shape[1] > max_addition_candidates:
            selection = torch.randperm(
                additions.shape[1], generator=generator
            )[:max_addition_candidates]
            additions = additions[:, selection]
        return {
            GraphOperation.ADD_EDGE: additions,
            GraphOperation.REMOVE_EDGE: current,
        }













    capped_sampling = bool(
        max_addition_candidates > 0
        and available_count > max_addition_candidates
        and rectangular_pair_count > max(1_000_000, 4 * max_addition_candidates)
    )
    if capped_sampling:
        target_count = min(max_addition_candidates, available_count)
        selected_codes: set[int] = set()
        selected_pairs: list[tuple[int, int]] = []
        while len(selected_pairs) < target_count:
            remaining = target_count - len(selected_pairs)
            draw_count = min(1_000_000, max(1024, 4 * remaining))
            flat_indices = torch.randint(
                rectangular_pair_count,
                (draw_count,),
                generator=generator,
            )
            sources = source_start + flat_indices // destination_count
            destinations = destination_start + flat_indices % destination_count
            for source, destination in zip(sources.tolist(), destinations.tolist()):
                if undirected and source >= destination:
                    continue
                if not undirected and not include_self and source == destination:
                    continue
                code = source * num_nodes + destination
                if code in current_legal_codes or code in selected_codes:
                    continue
                selected_codes.add(code)
                selected_pairs.append((source, destination))
                if len(selected_pairs) == target_count:
                    break
        additions = (
            torch.tensor(selected_pairs, dtype=torch.long).t().contiguous()
            if selected_pairs
            else torch.empty((2, 0), dtype=torch.long)
        )
    else:
        sources = torch.arange(source_start, source_start + source_count)
        destinations = torch.arange(
            destination_start, destination_start + destination_count
        )
        all_source = sources[:, None].expand(-1, destinations.numel()).reshape(-1)
        all_destination = destinations.repeat(sources.numel())
        if undirected:
            canonical = all_source.lt(all_destination)
            all_source = all_source[canonical]
            all_destination = all_destination[canonical]
        elif not include_self:
            non_self = all_source.ne(all_destination)
            all_source = all_source[non_self]
            all_destination = all_destination[non_self]
        all_codes = all_source * num_nodes + all_destination
        available = ~torch.isin(all_codes, current_codes)
        additions = torch.stack([all_source[available], all_destination[available]])
        if (
            max_addition_candidates > 0
            and additions.shape[1] > max_addition_candidates
        ):
            selection = torch.randperm(
                additions.shape[1], generator=generator
            )[:max_addition_candidates]
            additions = additions[:, selection]
    return {
        GraphOperation.ADD_EDGE: additions,
        GraphOperation.REMOVE_EDGE: (
            torch.tensor(
                [
                    [code // num_nodes for code in sorted(current_legal_codes)],
                    [code % num_nodes for code in sorted(current_legal_codes)],
                ],
                dtype=torch.long,
            )
            if undirected
            else current
        ),
    }


def _topology_pair_features(
    pair_state: CausalPairState | None,
    candidate_edges: dict[GraphOperation, torch.Tensor],
    *,
    time_index: int,
) -> dict[GraphOperation, torch.Tensor | None]:
    if pair_state is None:
        return {
            GraphOperation.ADD_EDGE: None,
            GraphOperation.REMOVE_EDGE: None,
        }
    return {
        GraphOperation.ADD_EDGE: pair_state.features(
            candidate_edges[GraphOperation.ADD_EDGE],
            time_index=time_index,
            currently_present=False,
        ),
        GraphOperation.REMOVE_EDGE: pair_state.features(
            candidate_edges[GraphOperation.REMOVE_EDGE],
            time_index=time_index,
            currently_present=True,
        ),
    }


def _topology_decoder_supervision_candidates(
    transition: dict[str, Any],
    action_candidates: dict[GraphOperation, torch.Tensor],
    *,
    num_nodes: int,
    node_weights: torch.Tensor,
    addition_neg_ratio: float,
    addition_negative_strategy: str,
    generator: torch.Generator,
) -> tuple[dict[GraphOperation, torch.Tensor], dict[str, torch.Tensor]]:

    if addition_neg_ratio < 0.0:
        raise ValueError("addition_neg_ratio must be non-negative.")
    if addition_negative_strategy not in {"uniform", "endpoint_corrupt", "mixed"}:
        raise ValueError(
            "addition_negative_strategy must be uniform, endpoint_corrupt, or mixed."
        )
    device = action_candidates[GraphOperation.ADD_EDGE].device
    target_add = _unique_edges(
        transition["edge_added"].to(device), num_nodes
    )
    target_remove = _unique_edges(
        transition["edge_removed"].to(device), num_nodes
    )
    positive_count = int(target_add.shape[1])
    requested_negatives = int(
        torch.ceil(
            torch.tensor(float(positive_count) * float(addition_neg_ratio))
        ).item()
    )
    full_add = action_candidates[GraphOperation.ADD_EDGE]
    full_codes = full_add[0].long() * num_nodes + full_add[1].long()
    positive_codes = target_add[0].long() * num_nodes + target_add[1].long()
    legal_negative = ~torch.isin(full_codes, positive_codes)
    selected_negative: list[torch.Tensor] = []

    def sample_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
        count = min(int(count), int(indices.numel()))
        if count <= 0:
            return indices[:0]
        permutation = torch.randperm(indices.numel(), generator=generator)[:count]
        return indices.index_select(0, permutation.to(indices.device))





    endpoint_requested = (
        requested_negatives
        if addition_negative_strategy == "endpoint_corrupt"
        else (requested_negatives + 1) // 2
        if addition_negative_strategy == "mixed"
        else 0
    )
    if endpoint_requested and target_add.numel():
        shares_endpoint = torch.isin(full_add[0], target_add[0]) | torch.isin(
            full_add[1], target_add[1]
        )
        endpoint_indices = torch.where(legal_negative & shares_endpoint)[0]
        selected = sample_indices(endpoint_indices, endpoint_requested)
        if selected.numel():
            selected_negative.append(selected)
            legal_negative[selected] = False

    selected_count = sum(int(index.numel()) for index in selected_negative)
    remaining = requested_negatives - selected_count
    if remaining > 0:
        uniform_indices = torch.where(legal_negative)[0]
        selected = sample_indices(uniform_indices, remaining)
        if selected.numel():
            selected_negative.append(selected)
    negative_index = (
        torch.cat(selected_negative)
        if selected_negative
        else full_add.new_empty((0,), dtype=torch.long)
    )
    negative_edges = full_add.index_select(1, negative_index)
    addition_edges = torch.cat([target_add, negative_edges], dim=1)
    addition_labels = torch.cat(
        [
            torch.ones(positive_count, device=device, dtype=torch.bool),
            torch.zeros(negative_edges.shape[1], device=device, dtype=torch.bool),
        ]
    )
    deletion_edges = action_candidates[GraphOperation.REMOVE_EDGE]
    deletion_codes = deletion_edges[0].long() * num_nodes + deletion_edges[1].long()
    remove_codes = target_remove[0].long() * num_nodes + target_remove[1].long()
    deletion_labels = torch.isin(deletion_codes, remove_codes)
    addition_weights = 0.5 * (
        node_weights[addition_edges[0]] + node_weights[addition_edges[1]]
    )
    deletion_weights = 0.5 * (
        node_weights[deletion_edges[0]] + node_weights[deletion_edges[1]]
    )
    return (
        {
            GraphOperation.ADD_EDGE: addition_edges,
            GraphOperation.REMOVE_EDGE: deletion_edges,
        },
        {
            "add_labels": addition_labels,
            "remove_labels": deletion_labels,
            "add_weights": addition_weights,
            "remove_weights": deletion_weights,
        },
    )


def _empty_edges(device: torch.device) -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _sequence_edges(
    sequence: GraphEditSequence,
    operation: GraphOperation,
    *,
    device: torch.device,
) -> torch.Tensor:
    targets = [edit.target for edit in sequence if edit.operation == operation]
    if not targets:
        return _empty_edges(device)
    return torch.tensor(targets, dtype=torch.long, device=device).t().contiguous()


def _sequence_nodes(
    sequence: GraphEditSequence,
    operation: GraphOperation,
    *,
    device: torch.device,
) -> torch.Tensor:
    nodes = [edit.target[0] for edit in sequence if edit.operation == operation]
    if not nodes:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(nodes, dtype=torch.long, device=device).unique()


def _topk_edges(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
) -> torch.Tensor:
    count = min(max(int(count), 0), int(candidates.shape[1]))
    if count == 0:
        return candidates[:, :0]
    return candidates[:, torch.topk(logits, k=count).indices]


def _topk_nodes(
    logits: torch.Tensor,
    count: int,
    *,
    candidates: torch.Tensor | None = None,
) -> torch.Tensor:
    candidates = (
        torch.arange(logits.numel(), device=logits.device)
        if candidates is None
        else candidates.to(device=logits.device, dtype=torch.long)
    )
    count = min(max(int(count), 0), int(candidates.numel()))
    if count == 0:
        return candidates[:0]
    return candidates[torch.topk(logits[candidates], k=count).indices]


def _topology_reward_context(
    transition: dict[str, Any],
    candidate_edges: dict[GraphOperation, torch.Tensor],
    node_weights: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    num_nodes = int(node_weights.shape[0])
    add_candidates = candidate_edges[GraphOperation.ADD_EDGE]
    remove_candidates = candidate_edges[GraphOperation.REMOVE_EDGE]
    target_add = transition["edge_added"].to(device)
    target_remove = transition["edge_removed"].to(device)
    target_add_codes = target_add[0] * num_nodes + target_add[1]
    target_remove_codes = target_remove[0] * num_nodes + target_remove[1]
    return {
        "add_labels": torch.isin(
            add_candidates[0] * num_nodes + add_candidates[1], target_add_codes
        ),
        "remove_labels": torch.isin(
            remove_candidates[0] * num_nodes + remove_candidates[1],
            target_remove_codes,
        ),
        "add_weights": 0.5
        * (node_weights[add_candidates[0]] + node_weights[add_candidates[1]]),
        "remove_weights": 0.5
        * (node_weights[remove_candidates[0]] + node_weights[remove_candidates[1]]),
    }


def _complete_topology_forecast_supervision(
    model: GraphWorldModel,
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    candidate_edges: dict[GraphOperation, torch.Tensor],
    node_weights: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    add_candidates = candidate_edges[GraphOperation.ADD_EDGE]
    target_add = transition["edge_added"].to(device)
    num_nodes = int(node_weights.shape[0])
    candidate_codes = add_candidates[0] * num_nodes + add_candidates[1]
    target_codes = target_add[0] * num_nodes + target_add[1]
    missing = ~torch.isin(target_codes, candidate_codes)
    if bool(missing.any()):
        missing_edges = target_add[:, missing]
        assert model.edge_addition_decoder is not None
        missing_logits = model.edge_addition_decoder(
            outputs["latent_next"][missing_edges[0]],
            outputs["latent_next"][missing_edges[1]],
        )
        add_candidates = torch.cat([add_candidates, missing_edges], dim=1)
        outputs = dict(outputs)
        outputs["edge_addition_logits"] = torch.cat(
            [outputs["edge_addition_logits"], missing_logits], dim=0
        )
    supervised_candidates = dict(candidate_edges)
    supervised_candidates[GraphOperation.ADD_EDGE] = add_candidates
    context = _topology_reward_context(
        transition,
        supervised_candidates,
        node_weights,
        device=device,
    )
    return outputs, context


def _proposed_node_value_reward(
    sequence: GraphEditSequence,
    *,
    operation: GraphOperation,
    target_delta: torch.Tensor,
    target_changed: torch.Tensor,
    node_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:

    changed = target_changed.to(device=device, dtype=torch.bool)
    prediction_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for edit in sequence:
        if edit.operation != operation or not edit.target or edit.value is None:
            continue
        node = int(edit.target[0])
        if node < 0 or node >= int(changed.numel()) or not bool(changed[node]):
            continue
        value = edit.value.to(device=device, dtype=target_delta.dtype).reshape(-1)
        target = target_delta[node].to(device=device, dtype=target_delta.dtype)
        if value.numel() != target.numel():
            raise ValueError(
                f"{operation.name} action value has {value.numel()} entries, "
                f"expected {target.numel()}."
            )
        prediction_rows.append(value)
        target_rows.append(target)
        weights.append(node_weights[node].to(device=device, dtype=target_delta.dtype))
    if not prediction_rows:
        zero = target_delta.new_zeros(())
        return {
            "reward": zero,
            "error": target_delta.new_full((), float("inf")),
            "rows": zero,
        }
    prediction = torch.stack(prediction_rows)
    target = torch.stack(target_rows)
    row_error = (prediction - target).abs().mean(dim=-1)
    row_weights = torch.stack(weights)
    error = (row_error * row_weights).sum() / row_weights.sum().clamp_min(1e-8)
    return {
        "reward": torch.exp(-error),
        "error": error,
        "rows": target_delta.new_tensor(float(len(prediction_rows))),
    }


def _decoded_reward(
    task: str,
    sequence: GraphEditSequence,
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    node_weights: torch.Tensor,
    *,
    candidate_edges: dict[GraphOperation, torch.Tensor],
    candidate_nodes: dict[GraphOperation, torch.Tensor],
    device: torch.device,
    topology_exact_reward_weight: float,
    topology_context: dict[str, torch.Tensor] | None,
    topology_target_z: torch.Tensor | None = None,
    topology_composite_embedding_reward: bool = True,
    topology_action_reward_weight: float = 0.0,
    node_action_reward_weight: float = 0.0,
    node_action_value_reward_weight: float = 0.0,
    property_mode: str = "logit_residual",
    property_change_threshold: float | None = None,
    property_reference_rows: torch.Tensor | None = None,
    property_reference_alpha: float = 1.0,
) -> tuple[float, dict[str, float]]:
    rollout_group = sequence.actions[0].operation
    if task != "topology" and rollout_group in {
        GraphOperation.ADD_EDGE,
        GraphOperation.REMOVE_EDGE,
    }:
        output_key = (
            "edge_addition_logits"
            if rollout_group == GraphOperation.ADD_EDGE
            else "edge_deletion_logits"
        )
        candidates = candidate_edges[rollout_group]
        target_edges = _sequence_edges(sequence, rollout_group, device=device)
        predicted_edges = _topk_edges(
            outputs[output_key], candidates, target_edges.shape[1]
        )
        metrics = structure_weighted_f1(
            predicted_edges,
            target_edges,
            node_weights=node_weights,
            item_type="edge",
        )
        return float(metrics["f1"].item()), {
            "synthetic_edge_fidelity_f1": float(metrics["f1"].item()),
            "synthetic_edge_fidelity_precision": float(
                metrics["precision"].item()
            ),
            "synthetic_edge_fidelity_recall": float(metrics["recall"].item()),
        }
    if task == "topology" and rollout_group == GraphOperation.MODIFY_NODE_STATE:
        target_nodes = _sequence_nodes(
            sequence, GraphOperation.MODIFY_NODE_STATE, device=device
        )
        predicted_nodes = torch.topk(
            outputs["node_change_logits"],
            k=min(target_nodes.numel(), outputs["node_change_logits"].numel()),
        ).indices
        localization = structure_weighted_f1(
            predicted_nodes,
            target_nodes,
            node_weights=node_weights,
            item_type="node",
        )
        assert "node_delta" in outputs
        if target_nodes.numel():
            delta_by_node = {
                edit.target[0]: edit.value
                for edit in sequence
                if edit.operation == GraphOperation.MODIFY_NODE_STATE
            }
            target_delta = torch.stack(
                [
                    delta_by_node[int(node)].to(
                        device=device, dtype=outputs["node_delta"].dtype
                    )
                    for node in target_nodes.tolist()
                ]
            )
            magnitude_error = F.smooth_l1_loss(
                outputs["node_delta"].index_select(0, target_nodes),
                target_delta,
            )
            magnitude_reward = torch.exp(-magnitude_error)
        else:
            magnitude_error = outputs["node_delta"].new_zeros(())
            magnitude_reward = outputs["node_delta"].new_ones(())
        combined = 0.5 * (localization["f1"] + magnitude_reward)
        return float(combined.item()), {
            "synthetic_state_fidelity_reward": float(combined.item()),
            "synthetic_state_localization_f1": float(
                localization["f1"].item()
            ),
            "synthetic_state_magnitude_error": float(magnitude_error.item()),
        }
    if task == "topology":
        add_candidates = candidate_edges[GraphOperation.ADD_EDGE]
        remove_candidates = candidate_edges[GraphOperation.REMOVE_EDGE]
        predicted_add = _topk_edges(
            outputs["edge_addition_logits"],
            add_candidates,
            sequence.operation_count(GraphOperation.ADD_EDGE),
        )
        predicted_remove = _topk_edges(
            outputs["edge_deletion_logits"],
            remove_candidates,
            sequence.operation_count(GraphOperation.REMOVE_EDGE),
        )
        result = topology_evolution_reward(
            predicted_add=predicted_add,
            target_add=transition["edge_added"].to(device),
            predicted_remove=predicted_remove,
            target_remove=transition["edge_removed"].to(device),
            node_weights=node_weights,
        )
        num_nodes = int(node_weights.shape[0])
        if topology_context is None:
            raise ValueError("Topology reward context is required for topology task.")
        dense_add = _balanced_dense_binary_reward(
            outputs["edge_addition_logits"],
            topology_context["add_labels"],
            topology_context["add_weights"],
        )
        dense_remove = _balanced_dense_binary_reward(
            outputs["edge_deletion_logits"],
            topology_context["remove_labels"],
            topology_context["remove_weights"],
        )
        dense_reward = 0.5 * (dense_add + dense_remove)
        exact_reward = result["reward"]
        edge_representation: dict[str, torch.Tensor] | None = None
        if topology_composite_embedding_reward and topology_target_z is not None:
            edge_representation = edge_representation_change_reward(
                predicted_node_next=outputs["latent_next"],
                current_node_embedding=outputs["z_t"],
                target_node_next=topology_target_z,
                current_edges=transition["edge_index_t"].to(device),
                next_edges=transition["edge_index_next"].to(device),
                node_weights=node_weights,
            )
            combined_reward = 0.25 * (
                result["addition_f1"]
                + result["deletion_f1"]
                + edge_representation["localization_f1"]
                + edge_representation["magnitude_reward"]
            )
        else:



            combined_reward = (
                topology_exact_reward_weight * exact_reward
                + (1.0 - topology_exact_reward_weight) * dense_reward
            )
        action_add = _sequence_edges(
            sequence, GraphOperation.ADD_EDGE, device=device
        )
        action_remove = _sequence_edges(
            sequence, GraphOperation.REMOVE_EDGE, device=device
        )
        def overlap_ratio(decoded: torch.Tensor, proposed: torch.Tensor) -> float:
            if decoded.shape[1] == 0:
                return 1.0 if proposed.shape[1] == 0 else 0.0
            decoded_codes = decoded[0] * num_nodes + decoded[1]
            proposed_codes = proposed[0] * num_nodes + proposed[1]
            return float(torch.isin(decoded_codes, proposed_codes).float().mean().item())








        direct_components: list[torch.Tensor] = []
        action_add_metrics: dict[str, torch.Tensor] | None = None
        action_remove_metrics: dict[str, torch.Tensor] | None = None
        if action_add.shape[1]:
            action_add_metrics = structure_weighted_f1(
                action_add,
                transition["edge_added"].to(device),
                node_weights=node_weights,
                item_type="edge",
            )
            direct_components.append(action_add_metrics["precision"])
        if action_remove.shape[1]:
            action_remove_metrics = structure_weighted_f1(
                action_remove,
                transition["edge_removed"].to(device),
                node_weights=node_weights,
                item_type="edge",
            )
            direct_components.append(action_remove_metrics["precision"])
        direct_action_reward = (
            torch.stack(direct_components).mean()
            if direct_components
            else combined_reward.new_zeros(())
        )
        combined_reward = (
            (1.0 - float(topology_action_reward_weight)) * combined_reward
            + float(topology_action_reward_weight) * direct_action_reward
        )
        details = {
            "combined_reward": float(combined_reward.item()),
            "exact_set_reward": float(exact_reward.item()),
            "dense_balanced_reward": float(dense_reward.item()),
            "dense_addition_reward": float(dense_add.item()),
            "dense_deletion_reward": float(dense_remove.item()),
            "addition_f1": float(result["addition_f1"].item()),
            "deletion_f1": float(result["deletion_f1"].item()),
            "decoder_action_add_overlap": overlap_ratio(predicted_add, action_add),
            "decoder_action_remove_overlap": overlap_ratio(
                predicted_remove, action_remove
            ),
            "controller_action_reward": float(direct_action_reward.item()),
            "controller_action_addition_precision": (
                float(action_add_metrics["precision"].item())
                if action_add_metrics is not None
                else None
            ),
            "controller_action_deletion_precision": (
                float(action_remove_metrics["precision"].item())
                if action_remove_metrics is not None
                else None
            ),
        }
        if edge_representation is not None:
            details.update(
                {
                    "edge_embedding_localization_f1": float(
                        edge_representation["localization_f1"].item()
                    ),
                    "edge_embedding_magnitude_reward": float(
                        edge_representation["magnitude_reward"].item()
                    ),
                    "edge_embedding_magnitude_error": float(
                        edge_representation["magnitude_error"].item()
                    ),
                }
            )
        return float(combined_reward.item()), details


    operation = (
        GraphOperation.MODIFY_NODE_PROPERTY
        if task == "node_property"
        else GraphOperation.MODIFY_NODE_STATE
    )
    count = sequence.operation_count(operation)
    eligible = candidate_nodes.get(operation)
    predicted_nodes = _topk_nodes(
        outputs["node_change_logits"], count, candidates=eligible
    )
    action_nodes = _sequence_nodes(sequence, operation, device=device)
    decoded_action_overlap = float(
        torch.isin(predicted_nodes, action_nodes).float().mean().item()
    ) if predicted_nodes.numel() else 1.0
    if task == "node_change":
        decoded_result = node_change_localization_reward(
            predicted_nodes=predicted_nodes,
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
        )
        action_result = node_change_localization_reward(
            predicted_nodes=action_nodes,
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
        )






        direct_action_reward = action_result["precision"]
        combined_reward = (
            (1.0 - float(node_action_reward_weight)) * decoded_result["reward"]
            + float(node_action_reward_weight) * direct_action_reward
        )
        return float(combined_reward.item()), {
            "localization_f1": float(decoded_result["f1"].item()),
            "controller_action_precision": float(action_result["precision"].item()),
            "controller_action_recall": float(action_result["recall"].item()),
            "controller_action_f1": float(action_result["f1"].item()),
            "controller_action_reward": float(direct_action_reward.item()),
            "decoder_action_node_overlap": decoded_action_overlap,
        }
    if task == "node_state":









        decoded_result = node_state_magnitude_reward(
            predicted_delta=outputs["node_delta"],
            target_delta=transition["node_delta"].to(device),
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
            selected_nodes=predicted_nodes,
        )
        action_result = node_change_localization_reward(
            predicted_nodes=action_nodes,
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
        )
        action_response = node_state_magnitude_reward(
            predicted_delta=outputs["node_delta"],
            target_delta=transition["node_delta"].to(device),
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
            selected_nodes=action_nodes,
        )
        action_value = _proposed_node_value_reward(
            sequence,
            operation=operation,
            target_delta=transition["node_delta"].to(device),
            target_changed=transition["node_changed"].to(device),
            node_weights=node_weights,
            device=device,
        )









        value_weight = float(node_action_value_reward_weight)
        response_or_value = (
            (1.0 - value_weight) * action_response["reward"]
            + value_weight * action_value["reward"]
        )
        direct_action_reward = 0.5 * (
            action_result["precision"] + response_or_value
        )
        combined_reward = (
            (1.0 - float(node_action_reward_weight)) * decoded_result["reward"]
            + float(node_action_reward_weight) * direct_action_reward
        )
        error = action_response["error"]
        finite_error = error if bool(torch.isfinite(error)) else error.new_tensor(1e12)
        return float(combined_reward.item()), {
            "magnitude_error": float(finite_error.item()),
            "decoder_action_node_overlap": decoded_action_overlap,
            "decoder_magnitude_reward": float(decoded_result["reward"].item()),
            "controller_action_precision": float(action_result["precision"].item()),
            "controller_action_recall": float(action_result["recall"].item()),
            "controller_action_f1": float(action_result["f1"].item()),
            "action_response_reward": float(action_response["reward"].item()),
            "action_response_magnitude_error": float(finite_error.item()),
            "action_value_reward": float(action_value["reward"].item()),
            "action_value_magnitude_error": float(
                action_value["error"].item()
                if bool(torch.isfinite(action_value["error"]))
                else 1e12
            ),
            "action_value_rows": float(action_value["rows"].item()),
        }

    prediction, _ = decode_property_prediction(
        outputs["node_property"],
        transition["property_t"].to(device),
        property_mode=property_mode,
        transition_gate=(
            torch.sigmoid(outputs["node_change_logits"])
            if property_mode in {"logit_mixture", "logit_residual_mixture"}
            else None
        ),
        current_observed_mask=(
            transition["property_observation_mask_t"].to(device).squeeze(-1).bool()
            if property_mode in {"logit_mixture", "logit_residual_mixture"}
            else None
        ),
    )
    property_node_ids = transition["property_node_ids"].to(device)
    if property_reference_rows is not None:
        raw_rows = prediction.index_select(0, property_node_ids)
        blended_rows = property_reference_rows.to(device) + float(
            property_reference_alpha
        ) * (raw_rows - property_reference_rows.to(device))
        prediction = prediction.clone()
        prediction.index_copy_(0, property_node_ids, blended_rows)
    result = node_property_reward(
        prediction=prediction.clamp_min(0.0),
        target=transition["property_target"].to(device),
        labelled_nodes=property_node_ids,
        node_weights=node_weights,
        mode="ranking",
        k=10,




        selected_nodes=None,
    )
    ranking_reward = result["reward"]
    action_result: dict[str, torch.Tensor] | None = None
    if property_change_threshold is not None and node_action_reward_weight > 0.0:






        target_changed = torch.zeros_like(node_weights, dtype=torch.float32)
        target_ids = transition["property_node_ids"].to(device=device, dtype=torch.long)
        comparable = transition["property_current_observed_mask"].to(
            device=device, dtype=torch.bool
        )
        magnitude = (
            transition["property_target"].to(device)
            - transition["property_current_target"].to(device)
        ).abs().mean(dim=-1)
        meaningful = magnitude.ge(float(property_change_threshold)) & comparable
        if target_ids.numel():
            target_changed.index_add_(0, target_ids, meaningful.to(torch.float32))
        action_result = node_change_localization_reward(
            predicted_nodes=action_nodes,
            target_changed=target_changed,
            node_weights=node_weights,
        )




        direct_action_reward = action_result["precision"]
        combined_reward = (
            (1.0 - float(node_action_reward_weight)) * ranking_reward
            + float(node_action_reward_weight) * direct_action_reward
        )
    else:
        combined_reward = ranking_reward
    details = {
        "ndcg_at_10": float(result.get("ndcg", result["reward"]).item()),
        "decoder_action_node_overlap": decoded_action_overlap,
    }
    if action_result is not None:
        details.update(
            {
                "controller_action_precision": float(
                    action_result["precision"].item()
                ),
                "controller_action_recall": float(action_result["recall"].item()),
                "controller_action_f1": float(action_result["f1"].item()),
                "controller_action_reward": float(direct_action_reward.item()),
            }
        )
    return float(combined_reward.item()), details


def _batched_topology_decoded_rewards(
    samples: list[Any],
    batched_outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    node_weights: torch.Tensor,
    *,
    candidate_edges: dict[GraphOperation, torch.Tensor],
    topology_context: dict[str, torch.Tensor],
    topology_exact_reward_weight: float,
    topology_target_z: torch.Tensor,
    topology_composite_embedding_reward: bool = True,
) -> torch.Tensor:
    if not samples:
        return node_weights.new_empty((0,))
    batch_size = len(samples)
    num_nodes = int(node_weights.shape[0])

    def dense_reward(logits: torch.Tensor, prefix: str) -> torch.Tensor:
        labels = topology_context[f"{prefix}_labels"].to(torch.bool)
        weights = topology_context[f"{prefix}_weights"].to(logits)
        probability = torch.sigmoid(logits)
        positive_weight = weights * labels.to(weights.dtype)
        negative_weight = weights * (~labels).to(weights.dtype)
        positive_denominator = positive_weight.sum()
        negative_denominator = negative_weight.sum()
        positive = (
            (probability * positive_weight.unsqueeze(0)).sum(dim=1)
            / positive_denominator.clamp_min(1e-8)
        )
        negative = (
            ((1.0 - probability) * negative_weight.unsqueeze(0)).sum(dim=1)
            / negative_denominator.clamp_min(1e-8)
        )
        present = torch.stack(
            [positive_denominator.gt(0), negative_denominator.gt(0)]
        ).to(logits.dtype)
        return (
            positive * present[0] + negative * present[1]
        ) / present.sum().clamp_min(1.0)

    def exact_f1(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        labels: torch.Tensor,
        target: torch.Tensor,
        operation: GraphOperation,
    ) -> torch.Tensor:
        counts_list = [
            sample.action.operation_count(operation) for sample in samples
        ]
        counts = torch.tensor(counts_list, device=logits.device, dtype=torch.long)
        maximum = max(counts_list, default=0)
        if maximum:
            indices = torch.topk(logits, k=maximum, dim=1).indices
            selected = (
                torch.arange(maximum, device=logits.device).unsqueeze(0)
                < counts.unsqueeze(1)
            )
            candidate_weights = 0.5 * (
                node_weights[candidates[0]] + node_weights[candidates[1]]
            )
            selected_weights = candidate_weights[indices]
            selected_correct = labels.to(torch.bool)[indices]
            predicted_weight = (selected_weights * selected).sum(dim=1)
            true_positive = (
                selected_weights * selected * selected_correct
            ).sum(dim=1)
        else:
            predicted_weight = logits.new_zeros((batch_size,))
            true_positive = logits.new_zeros((batch_size,))

        if target.numel():
            target_codes = torch.unique(
                target[0].long() * num_nodes + target[1].long()
            )
            target_weight = 0.5 * (
                node_weights[target_codes // num_nodes]
                + node_weights[target_codes % num_nodes]
            ).sum()
        else:
            target_weight = logits.new_zeros(())
        precision = true_positive / predicted_weight.clamp_min(1e-8)
        recall = true_positive / target_weight.clamp_min(1e-8)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        if not target.numel():
            f1 = torch.where(counts.eq(0), torch.ones_like(f1), f1)
        return f1

    target_add = transition["edge_added"].to(node_weights.device)
    target_remove = transition["edge_removed"].to(node_weights.device)
    add_logits = batched_outputs["edge_addition_logits"]
    remove_logits = batched_outputs["edge_deletion_logits"]
    add_f1 = exact_f1(
        add_logits,
        candidate_edges[GraphOperation.ADD_EDGE],
        topology_context["add_labels"],
        target_add,
        GraphOperation.ADD_EDGE,
    )
    remove_f1 = exact_f1(
        remove_logits,
        candidate_edges[GraphOperation.REMOVE_EDGE],
        topology_context["remove_labels"],
        target_remove,
        GraphOperation.REMOVE_EDGE,
    )
    if not topology_composite_embedding_reward:
        exact_reward = 0.5 * (add_f1 + remove_f1)
        dense_add = dense_reward(add_logits, "add")
        dense_remove = dense_reward(remove_logits, "remove")
        dense_combined = 0.5 * (dense_add + dense_remove)
        return (
            float(topology_exact_reward_weight) * exact_reward
            + (1.0 - float(topology_exact_reward_weight)) * dense_combined
        )
    edge_representation = edge_representation_change_reward(
        predicted_node_next=batched_outputs["latent_next"],
        current_node_embedding=batched_outputs["z_t"][0],
        target_node_next=topology_target_z,
        current_edges=transition["edge_index_t"].to(node_weights.device),
        next_edges=transition["edge_index_next"].to(node_weights.device),
        node_weights=node_weights,
    )
    return 0.25 * (
        add_f1
        + remove_f1
        + edge_representation["localization_f1"]
        + edge_representation["magnitude_reward"]
    )


def _binary_action_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive = labels.to(torch.bool)
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        negative_logits = logits[negative]
        positive_count = int(positive.sum().item())
        hard_count = min(
            int(negative_logits.numel()),
            max(32, 32 * max(positive_count, 1)),
        )
        hard_negative_logits = torch.topk(negative_logits, k=hard_count).values
        terms.append(F.softplus(hard_negative_logits).mean())
        if bool(positive.any()):



            terms.append(
                F.softplus(
                    hard_negative_logits.max() - logits[positive].mean()
                )
            )
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean()


def _topology_controller_supervision_loss(
    controller: ActionAwareController,
    node_state: torch.Tensor,
    candidate_edges: dict[GraphOperation, torch.Tensor],
    topology_context: dict[str, torch.Tensor],
    transition: dict[str, Any],
    *,
    count_weight: float = 0.0,
    count_target_mode: str = "absolute",
    count_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count_weight < 0.0:
        raise ValueError("count_weight must be non-negative.")
    if count_target_mode not in {"absolute", "edge_rate"}:
        raise ValueError("count_target_mode must be absolute or edge_rate.")
    losses: list[torch.Tensor] = []
    for operation, label_key in (
        (GraphOperation.ADD_EDGE, "add_labels"),
        (GraphOperation.REMOVE_EDGE, "remove_labels"),
    ):
        candidates = candidate_edges[operation]
        labels = topology_context[label_key].to(torch.bool)











        if operation == GraphOperation.ADD_EDGE:
            num_nodes = int(node_state.shape[0])
            target_additions = _unique_edges(
                transition["edge_added"].to(
                    device=candidates.device, dtype=torch.long
                ),
                num_nodes,
            )
            if target_additions.shape[1]:
                candidate_codes = candidates[0] * num_nodes + candidates[1]
                target_codes = (
                    target_additions[0] * num_nodes + target_additions[1]
                )
                missing = ~torch.isin(target_codes, candidate_codes)
                if bool(missing.any()):
                    candidates = torch.cat(
                        [candidates, target_additions[:, missing]], dim=1
                    )
                    labels = torch.cat(
                        [
                            labels,
                            torch.ones(
                                int(missing.sum().item()),
                                dtype=torch.bool,
                                device=labels.device,
                            ),
                        ]
                    )
        positive_index = torch.where(labels)[0]
        negative_index = torch.where(~labels)[0]
        negative_budget = min(
            int(negative_index.numel()),
            max(256, min(8192, 32 * max(int(positive_index.numel()), 1))),
        )
        if negative_index.numel() > negative_budget:
            evenly_spaced = torch.linspace(
                0,
                negative_index.numel() - 1,
                steps=negative_budget,
                device=negative_index.device,
            ).long()
            negative_index = negative_index[evenly_spaced]
        selected = torch.cat([positive_index, negative_index])
        if selected.numel() == 0:
            continue
        selected_edges = candidates.index_select(1, selected)
        selected_labels = labels.index_select(0, selected).to(node_state.dtype)
        selected_logits = controller.edge_logits_chunked(
            node_state, selected_edges, chunk_size=65536
        )
        losses.append(_binary_action_loss(selected_logits, selected_labels))
    pair_loss = (
        torch.stack(losses).mean() if losses else node_state.new_zeros(())
    )
    if count_weight > 0.0:
        denominator = 1.0
        if count_target_mode == "edge_rate":
            denominator = float(transition["edge_index_t"].shape[1])
            denominator = max(denominator, 1.0)
        target_log_count = torch.stack(
            [
                torch.log1p(
                    node_state.new_tensor(
                        float(transition["edge_added"].shape[1]) / denominator
                    )
                ),
                torch.log1p(
                    node_state.new_tensor(
                        float(transition["edge_removed"].shape[1]) / denominator
                    )
                ),
            ]
        )
        count_loss = F.smooth_l1_loss(
            controller.topology_log_counts(node_state, context=count_context),
            target_log_count,
        )
    else:
        count_loss = node_state.new_zeros(())
    return pair_loss, count_loss


def _balanced_dense_binary_reward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    positive = labels.to(torch.bool)
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        positive_weight = weights[positive]
        terms.append(
            (probability[positive] * positive_weight).sum()
            / positive_weight.sum().clamp_min(1e-8)
        )
    if bool(negative.any()):
        negative_weight = weights[negative]
        terms.append(
            ((1.0 - probability[negative]) * negative_weight).sum()
            / negative_weight.sum().clamp_min(1e-8)
        )
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean()


def _observable_intervention_loss(
    task: str,
    sequence: GraphEditSequence,
    outputs: dict[str, torch.Tensor],
    intervention: dict[str, torch.Tensor | None],
    transition: dict[str, Any],
    *,
    candidate_edges: dict[GraphOperation, torch.Tensor],
    device: torch.device,
    property_mode: str = "delta",
) -> torch.Tensor:
    rollout_group = sequence.actions[0].operation
    if rollout_group in {
        GraphOperation.ADD_EDGE,
        GraphOperation.REMOVE_EDGE,
    }:
        losses: list[torch.Tensor] = []
        for operation, output_key in (
            (GraphOperation.ADD_EDGE, "edge_addition_logits"),
            (GraphOperation.REMOVE_EDGE, "edge_deletion_logits"),
        ):
            if operation != rollout_group and task != "topology":
                continue
            candidates = candidate_edges[operation]
            candidate_codes = candidates[0] * transition["x_t"].shape[0] + candidates[1]
            action_edges = _sequence_edges(sequence, operation, device=device)
            action_codes = (
                action_edges[0] * transition["x_t"].shape[0] + action_edges[1]
                if action_edges.numel()
                else torch.empty(0, dtype=torch.long, device=device)
            )
            labels = torch.isin(candidate_codes, action_codes).to(outputs[output_key].dtype)
            losses.append(_binary_action_loss(outputs[output_key], labels))
        return torch.stack(losses).mean()

    operation = rollout_group
    nodes = _sequence_nodes(sequence, operation, device=device)
    labels = outputs["node_change_logits"].new_zeros(
        outputs["node_change_logits"].shape
    )
    labels[nodes] = 1.0
    localization = _binary_action_loss(outputs["node_change_logits"], labels)
    assert intervention["x"] is not None
    synthetic_delta = intervention["x"] - transition["x_t"]
    if task == "node_change":
        return localization
    if operation == GraphOperation.MODIFY_NODE_STATE:
        magnitude = F.smooth_l1_loss(outputs["node_delta"][nodes], synthetic_delta[nodes])
        return localization + magnitude
    property_slice = transition["property_slice"]
    current_property = transition["x_t"][nodes, property_slice]
    target_property = intervention["x"][nodes, property_slice]
    if property_mode in {"logit_mixture", "logit_residual_mixture"}:
        raise ValueError(
            "Synthetic property interventions currently require a non-mixture "
            "property mode because their decoder gate is not part of the "
            "intervention target path."
        )
    predicted_property, _ = decode_property_prediction(
        outputs["node_property"][nodes],
        current_property,
        property_mode=property_mode,
    )
    magnitude = F.smooth_l1_loss(predicted_property, target_property)
    return localization + magnitude


def _observed_transition_loss(
    task: str,
    outputs: dict[str, torch.Tensor],
    transition: dict[str, Any],
    *,
    target_z: torch.Tensor | None,
    topology_context: dict[str, torch.Tensor] | None,
    property_mode: str,
    latent_weight: float,
    change_weight: float,
    change_pos_weight: float,
    state_weight: float,
    state_loss_mode: str,
    state_loss_type: str,
    property_loss_type: str,
    property_change_threshold: float | None,
    latent_cosine_weight: float,
    latent_variance_weight: float,
    all_state_weight: float,
    current_state_weight: float,
    topology_count_weight: float,
    topology_decoder_loss: str,
    device: torch.device,
    action_plan_target: torch.Tensor | None = None,
    action_plan_consistency_weight: float = 0.0,
    action_plan_positive_weight_cap: float = 0.0,
    action_decoder_consistency_weight: float = 0.0,
    action_decoder_positive_weight_cap: float = 0.0,
    action_value_decoder_consistency_weight: float = 0.0,
    action_value_target: torch.Tensor | None = None,
    action_value_probability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:

    zero = outputs["latent_next"].new_zeros(())
    latent = (
        latent_transition_terms(
            outputs,
            target_z,
            cosine_weight=latent_cosine_weight,
            variance_weight=latent_variance_weight,
        )["total"]
        if target_z is not None
        else zero
    )
    change = zero
    observable = zero
    current_state = zero
    action_plan = zero
    action_decoder_consistency = zero
    action_value_decoder_consistency = zero
    if task == "topology":
        if topology_context is None:
            raise ValueError("Topology forecast supervision requires labels.")
        if topology_decoder_loss not in {"bce", "balanced_hard"}:
            raise ValueError("topology_decoder_loss must be bce or balanced_hard.")
        binary_loss = (
            F.binary_cross_entropy_with_logits
            if topology_decoder_loss == "bce"
            else _binary_action_loss
        )
        add_labels = topology_context["add_labels"].to(
            dtype=outputs["edge_addition_logits"].dtype
        )
        remove_labels = topology_context["remove_labels"].to(
            dtype=outputs["edge_deletion_logits"].dtype
        )
        edge_terms: list[torch.Tensor] = []
        if add_labels.numel():
            edge_terms.append(
                binary_loss(outputs["edge_addition_logits"], add_labels)
            )
        if remove_labels.numel():
            edge_terms.append(
                binary_loss(outputs["edge_deletion_logits"], remove_labels)
            )
        edge_transition = (
            torch.stack(edge_terms).mean() if edge_terms else zero
        )
        observable = edge_transition
        if topology_count_weight:
            if "topology_count_log1p" not in outputs:
                raise ValueError(
                    "A positive topology count weight requires the count decoder."
                )
            count_target = torch.log1p(
                torch.stack(
                    [
                        topology_context["add_labels"].sum(),
                        topology_context["remove_labels"].sum(),
                    ]
                ).to(outputs["topology_count_log1p"])
            )
            count_loss = F.smooth_l1_loss(
                outputs["topology_count_log1p"], count_target
            )
            observable = observable + float(topology_count_weight) * count_loss
    elif task in {"node_change", "node_state"}:
        labels = transition["node_changed"].to(
            device=device, dtype=outputs["node_change_logits"].dtype
        )
        change = F.binary_cross_entropy_with_logits(
            outputs["node_change_logits"],
            labels,
            pos_weight=outputs["node_change_logits"].new_tensor(change_pos_weight),
        )






        if task == "node_state" or state_weight > 0.0:
            target_delta = transition["node_delta"].to(device)
            changed = labels.bool()
            state_loss = F.l1_loss if state_loss_type == "l1" else F.smooth_l1_loss
            if state_loss_mode in {"changed_only", "conditional_changed"} and bool(changed.any()):
                state_prediction = (
                    outputs["node_delta_ungated"]
                    if state_loss_mode == "conditional_changed"
                    else outputs["node_delta"]
                )
                conditional = state_loss(
                    state_prediction[changed], target_delta[changed]
                )
                observable = conditional
                if state_loss_mode == "conditional_changed":
                    all_nodes = state_loss(outputs["node_delta"], target_delta)
                    observable = observable + float(all_state_weight) * all_nodes
            else:
                observable = state_loss(outputs["node_delta"], target_delta)
            if current_state_weight > 0.0:
                if "x_t_reconstructed" not in outputs:
                    raise ValueError(
                        "A positive current-state weight requires the current-state decoder."
                    )
                current_state = F.smooth_l1_loss(
                    outputs["x_t_reconstructed"], transition["x_t"].to(device)
                )
    else:




        if change_weight > 0.0:
            if property_change_threshold is None:
                raise ValueError(
                    "A positive property change weight requires a train-only threshold."
                )
            property_rows = transition["property_node_ids"].to(device=device, dtype=torch.long)
            comparable = transition["property_current_observed_mask"].to(
                device=device, dtype=torch.bool
            )
            property_delta = (
                transition["property_target"].to(device)
                - transition["property_current_target"].to(device)
            )
            labels = property_delta.abs().mean(dim=-1).ge(
                float(property_change_threshold)
            ).to(dtype=outputs["node_change_logits"].dtype)
            logits = outputs["node_change_logits"].index_select(0, property_rows)
            logits = logits[comparable]
            labels = labels[comparable]
            if labels.numel():
                positives = labels.sum()
                negatives = labels.numel() - positives
                if property_mode in {"logit_mixture", "logit_residual_mixture"}:






                    change = F.binary_cross_entropy_with_logits(logits, labels)
                else:
                    pos_weight = (
                        negatives / positives.clamp_min(1.0)
                    ).clamp(1.0, 20.0)
                    change = F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight
                    )
        prediction, _ = decode_property_prediction(
            outputs["node_property"],
            transition["property_t"].to(device),
            property_mode=property_mode,
            transition_gate=(
                torch.sigmoid(outputs["node_change_logits"])
                if property_mode in {"logit_mixture", "logit_residual_mixture"}
                else None
            ),
            current_observed_mask=(
                transition["property_observation_mask_t"].to(device).squeeze(-1).bool()
                if property_mode in {"logit_mixture", "logit_residual_mixture"}
                else None
            ),
        )
        rows = transition["property_node_ids"].to(device)
        target = transition["property_target"].to(device)
        prediction_rows = prediction.index_select(0, rows)
        use_cross_entropy = property_loss_type == "cross_entropy" or (
            property_loss_type == "auto" and property_mode.startswith("logit_")
        )
        if use_cross_entropy:



            normalized_target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            observable = -(
                normalized_target * prediction_rows.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        elif property_loss_type == "mse":
            observable = F.mse_loss(prediction_rows, target)
        else:
            observable = F.smooth_l1_loss(prediction_rows, target)

    if action_plan_consistency_weight > 0.0:
        if action_plan_target is None or "action_plan_logits" not in outputs:
            raise ValueError(
                "A positive action-plan consistency weight requires a "
                "past-only target and action_plan_logits."
            )
        target = action_plan_target.detach().to(
            device=device, dtype=outputs["action_plan_logits"].dtype
        )
        if target.shape != outputs["action_plan_logits"].shape:
            raise ValueError(
                "action_plan_target must match action_plan_logits, got "
                f"{tuple(target.shape)} and "
                f"{tuple(outputs['action_plan_logits'].shape)}."
            )





        if action_plan_positive_weight_cap > 0.0:





            reduction_dims = tuple(range(target.ndim - 1)) if target.ndim > 1 else (0,)
            positive_mass = target.sum(dim=reduction_dims)
            negative_mass = (1.0 - target).sum(dim=reduction_dims)
            pos_weight = (
                negative_mass / positive_mass.clamp_min(1e-6)
            ).clamp(1.0, float(action_plan_positive_weight_cap))
            action_plan = F.binary_cross_entropy_with_logits(
                outputs["action_plan_logits"], target, pos_weight=pos_weight
            )
        else:
            action_plan = F.binary_cross_entropy_with_logits(
                outputs["action_plan_logits"], target
            )

    if action_decoder_consistency_weight > 0.0:
        if task not in {"node_change", "node_state", "node_property"}:
            raise ValueError(
                "Direct action-decoder consistency is currently defined only "
                "for node-edit tasks."
            )
        if action_plan_target is None:
            raise ValueError(
                "A positive action-decoder consistency weight requires a "
                "detached, past-only Controller action target."
            )
        target = action_plan_target.detach().to(
            device=device, dtype=outputs["node_change_logits"].dtype
        )
        logits = outputs["node_change_logits"]
        if target.shape != logits.shape:
            raise ValueError(
                "Node-action target must match the future-latent change "
                f"decoder shape, got {tuple(target.shape)} and "
                f"{tuple(logits.shape)}."
            )




        if action_decoder_positive_weight_cap > 0.0:





            positive_mass = target.sum()
            negative_mass = (1.0 - target).sum()
            pos_weight = (
                negative_mass / positive_mass.clamp_min(1e-6)
            ).clamp(1.0, float(action_decoder_positive_weight_cap))
            action_decoder_consistency = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
        else:
            action_decoder_consistency = F.binary_cross_entropy_with_logits(
                logits, target
            )

    if action_value_decoder_consistency_weight > 0.0:
        if task != "node_state":
            raise ValueError(
                "Action-value decoder consistency is currently defined only "
                "for node-state transitions."
            )
        if action_value_target is None or action_value_probability is None:
            raise ValueError(
                "A positive action-value decoder consistency weight requires "
                "a past-only Controller magnitude and node proposal probability."
            )
        if "node_delta" not in outputs:
            raise ValueError(
                "Action-value decoder consistency requires the future-latent "
                "node-state decoder."
            )
        value_target = action_value_target.detach().to(
            device=device, dtype=outputs["node_delta"].dtype
        )
        probability = action_value_probability.detach().to(
            device=device, dtype=outputs["node_delta"].dtype
        )
        if value_target.shape != outputs["node_delta"].shape:
            raise ValueError(
                "Controller magnitude must match node_delta, got "
                f"{tuple(value_target.shape)} and "
                f"{tuple(outputs['node_delta'].shape)}."
            )
        if probability.shape != outputs["node_delta"].shape[:1]:
            raise ValueError(
                "Controller node proposal probability must have shape "
                f"{tuple(outputs['node_delta'].shape[:1])}, got "
                f"{tuple(probability.shape)}."
            )







        per_node_error = F.smooth_l1_loss(
            outputs["node_delta"], value_target, reduction="none"
        ).mean(dim=-1)
        weight = probability.clamp_min(0.0)
        action_value_decoder_consistency = (
            per_node_error * weight
        ).sum() / weight.sum().clamp_min(1e-6)

    total = (
        float(latent_weight) * latent
        + float(change_weight) * change
        + float(state_weight) * observable
        + float(current_state_weight) * current_state
        + float(action_plan_consistency_weight) * action_plan
        + float(action_decoder_consistency_weight) * action_decoder_consistency
        + float(action_value_decoder_consistency_weight)
        * action_value_decoder_consistency
    )
    return {
        "total": total,
        "latent": latent,
        "change": change,
        "observable": observable,
        "current_state": current_state,
        "action_plan": action_plan,
        "action_decoder_consistency": action_decoder_consistency,
        "action_value_decoder_consistency": action_value_decoder_consistency,
    }


def _latent_action_locality_loss(
    outputs: dict[str, torch.Tensor],
    sequence: GraphEditSequence,
    *,
    num_nodes: int,
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    target_nodes = sorted(
        {node for edit in sequence for node in edit.target}
    )
    if not target_nodes or len(target_nodes) >= num_nodes:
        return outputs["latent_next"].new_zeros(())
    target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    target_mask[torch.tensor(target_nodes, dtype=torch.long, device=device)] = True
    latent_change = (outputs["latent_next"] - outputs["z_t"]).pow(2).mean(dim=-1).sqrt()
    target_change = latent_change[target_mask].mean()
    background_change = latent_change[~target_mask].mean()
    return F.relu(float(margin) + background_change - target_change)


def _select_diverse_world_rollouts(
    rollouts: list[RewardedSequenceRollout],
    budget: int,
) -> list[RewardedSequenceRollout]:
    budget = max(1, min(int(budget), len(rollouts)))
    selected: list[RewardedSequenceRollout] = []
    selected_ids: set[int] = set()

    def add(rollout: RewardedSequenceRollout) -> None:
        identity = id(rollout)
        if identity not in selected_ids and len(selected) < budget:
            selected.append(rollout)
            selected_ids.add(identity)

    grouped: dict[GraphOperation, list[RewardedSequenceRollout]] = {}
    for rollout in rollouts:
        first_operation = rollout.sample.action.actions[0].operation
        grouped.setdefault(first_operation, []).append(rollout)
    for group in grouped.values():
        add(max(group, key=lambda rollout: rollout.reward))
    for group in grouped.values():
        add(group[0])
        add(group[-1])
    for rollout in sorted(rollouts, key=lambda item: item.reward, reverse=True):
        add(rollout)
    return selected


def _past_change_update(
    task: str,
    transition: dict[str, Any],
    device: torch.device,
    *,
    num_nodes: int | None = None,
) -> torch.Tensor:
    if task == "topology":
        update = torch.zeros(
            transition["x_t"].shape[0], device=device, dtype=torch.float32
        )
        changed_edges = torch.cat(
            [transition["edge_added"], transition["edge_removed"]], dim=1
        ).to(device=device, dtype=torch.long)
        if changed_edges.numel():
            endpoints = torch.unique(changed_edges.reshape(-1))
            update.index_fill_(0, endpoints, 1.0)
        return update
    if task == "node_property":
        update_size = (
            int(num_nodes)
            if num_nodes is not None
            else int(transition["property_t"].shape[0])
        )
        update = torch.zeros(
            update_size, device=device, dtype=torch.float32
        )
        comparable = transition["property_current_observed_mask"].to(
            device=device, dtype=torch.bool
        )
        target_ids = transition["property_node_ids"].to(device=device, dtype=torch.long)
        changed = (
            transition["property_target"].to(device)
            - transition["property_current_target"].to(device)
        ).abs().amax(dim=-1).gt(1e-8) & comparable
        if target_ids.numel():
            update.index_add_(0, target_ids, changed.to(torch.float32))
        return update
    return transition["node_changed"].to(device=device, dtype=torch.float32)


def _past_change_observation_count(
    task: str,
    transition: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if task == "topology":
        return torch.zeros((), device=device, dtype=torch.float32)
    if task == "node_property":
        comparable = transition["property_current_observed_mask"].to(
            device=device, dtype=torch.float32
        )
        return comparable.sum()
    return torch.tensor(
        float(transition["node_changed"].numel()),
        device=device,
        dtype=torch.float32,
    )


def _observed_operation_groups(
    task: str,
    transition: dict[str, Any],
) -> tuple[GraphOperation, ...]:
    if task == "topology":
        additions = int(transition["edge_added"].shape[1])
        removals = int(transition["edge_removed"].shape[1])
        return (
            *(GraphOperation.ADD_EDGE for _ in range(additions)),
            *(GraphOperation.REMOVE_EDGE for _ in range(removals)),
        )
    if task in {"node_change", "node_state"}:
        changed = int(transition["node_changed"].to(torch.bool).sum().item())
        return tuple(GraphOperation.MODIFY_NODE_STATE for _ in range(changed))
    if task == "node_property":
        current = transition["property_current_target"].float()
        target = transition["property_target"].float()
        comparable = transition["property_current_observed_mask"].to(torch.bool)
        changed = (target - current).abs().amax(dim=-1).gt(1e-8) & comparable
        return tuple(
            GraphOperation.MODIFY_NODE_PROPERTY
            for _ in range(int(changed.sum().item()))
        )
    raise ValueError(f"Unsupported task={task!r} for operation history.")


def _rotating_clip_indices(
    total: int,
    budget: int,
    epoch_index: int,
    epochs: int,
) -> range:

    if total < 1:
        return range(0)
    effective_budget = total if budget == 0 else min(int(budget), int(total))
    if effective_budget >= total:
        return range(total)
    maximum_start = int(total) - effective_budget
    denominator = max(int(epochs) - 1, 1)
    start = round(float(epoch_index) * float(maximum_start) / denominator)
    return range(int(start), int(start) + effective_budget)


def _restore_group_sampler_prefix(
    sampler: DynamicGroupSampler,
    executed_operations_by_transition: dict[
        int, tuple[GraphOperation, ...]
    ],
    transition_index: int,
    fallback_operations_by_transition: dict[
        int, tuple[GraphOperation, ...]
    ] | None = None,
) -> None:

    sampler.counts.clear()
    missing: list[int] = []
    for index in range(int(transition_index)):
        operations = executed_operations_by_transition.get(index)
        if operations is None and fallback_operations_by_transition is not None:
            operations = fallback_operations_by_transition.get(index)
        if operations is None:
            missing.append(index)
        else:
            sampler.observe(operations)
    if missing:
        first = missing[0]
        last = missing[-1]
        raise RuntimeError(
            "Cannot restore the full observed-action prefix before "
            f"transition {transition_index}: missing cached transitions "
            f"{first}..{last}. Increase --steps so adjacent rotating clips "
            "overlap, or use --steps 0."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["topology", "node_change", "node_state", "node_property"],
        default="node_change",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Canonical dataset alias; defaults to the first registered dataset for the task.",
    )
    parser.add_argument("--processed", default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help=(
            "Chronological transitions updated per epoch.  When smaller than "
            "the train split, epochs use deterministic rotating contiguous "
            "clips so that temporal order is preserved and the full timeline "
            "is covered across training; 0 uses the complete train split."
        ),
    )
    parser.add_argument(
        "--clip_history_burn_in",
        type=int,
        default=32,
        help=(
            "At the beginning of a rotating training clip, replay at most "
            "this many immediately preceding observed source snapshots to "
            "warm-start H and the state-memory cache; 0 keeps a cold reset."
        ),
    )
    parser.add_argument(
        "--clip_history_action_replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When a bounded rotating clip starts in T2/T3 proposal mode, "
            "replay past-only Controller proposals together with observed "
            "source graphs so the action-memory cache matches evaluation."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--val_every",
        type=int,
        default=0,
        help="Run future-free validation every N epochs; 0 disables validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument("--rollout_budget", type=int, default=16)
    parser.add_argument(
        "--batch_topology_grpo_rollouts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Batch mixed ADD_EDGE/REMOVE_EDGE Controller sampling and GRPO "
            "likelihood evaluation without changing rollout allocation or "
            "action-sequence semantics; enabled by default for topology."
        ),
    )
    parser.add_argument(
        "--batch_topology_grpo_rewards",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Batch topology decoder-only structure rewards on-device; enabled "
            "by default for topology and disabled for the other tasks."
        ),
    )
    parser.add_argument(
        "--topology_composite_embedding_reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include edge-representation localization and magnitude in the "
            "T2 verifier reward. Disable only for controlled comparison with "
            "the former exact/dense topology reward."
        ),
    )
    parser.add_argument(
        "--cache_deterministic_topology_candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache on CPU only topology candidate sets that were constructed "
            "without random truncation. Disabled by default for controlled "
            "timing comparison."
        ),
    )
    parser.add_argument(
        "--cache_device_topology_candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache complete deterministic topology action candidates and "
            "their causal pair features on the training device. Randomly "
            "truncated candidate sets are never cached. Disabled by default."
        ),
    )
    parser.add_argument("--world_intervention_rollouts", type=int, default=8)
    parser.add_argument(
        "--sample_action_magnitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether sampled unary Controller rollouts include a continuous "
            "state/property magnitude. Disable this when the forecast action "
            "is intentionally target-only, so the GRPO rollout and inference "
            "action spaces remain identical."
        ),
    )
    parser.add_argument(
        "--grpo_update_interval",
        type=int,
        default=1,
        help=(
            "Sample Controller groups and apply GRPO once every N chronological "
            "world-model updates. Forecast and supervised Controller updates "
            "still use every transition. This decouples graph-state coverage "
            "from the expensive rollout frequency."
        ),
    )
    parser.add_argument("--min_actions", type=int, default=1)
    parser.add_argument("--max_actions", type=int, default=8)
    parser.add_argument(
        "--inference_stop_threshold",
        type=float,
        default=0.5,
        help=(
            "Future-free greedy STOP threshold used for the single action "
            "sequence committed to recurrent history."
        ),
    )
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument("--max_addition_candidates", type=int, default=100000)
    parser.add_argument(
        "--topology_addition_neg_ratio",
        type=float,
        default=1.0,
        help=(
            "For the topology decoder only, number of sampled legal "
            "addition negatives per true addition. Controller proposals are "
            "still drawn from the full current-state legal pair space."
        ),
    )
    parser.add_argument(
        "--topology_addition_negative_strategy",
        choices=["uniform", "endpoint_corrupt", "mixed"],
        default="uniform",
        help=(
            "Negative construction for the topology decoder after the "
            "future-free action proposal is fixed."
        ),
    )
    parser.add_argument(
        "--topology_decoder_loss",
        choices=["bce", "balanced_hard"],
        default="bce",
        help=(
            "Topology observable loss. BCE matches the established passive "
            "topology-transition trainer; balanced_hard is the older action "
            "prototype loss."
        ),
    )
    parser.add_argument(
        "--topology_pair_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition T1 Delta-E decoders on a sparse causal pair state "
            "(past count, recency, persistence and current membership)."
        ),
    )
    parser.add_argument(
        "--topology_candidate_policy",
        choices=["all_pairs", "seen_pairs"],
        default="all_pairs",
        help=(
            "Addition candidate universe. seen_pairs is a causal reactivation "
            "track using only completed source snapshots."
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument(
        "--separate_action_query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject action through a separate zero-initialized query residual, "
            "preserving the passive Transformer's query fan-in."
        ),
    )
    parser.add_argument(
        "--bounded_action_residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Start from an exact passive state transition and learn a "
            "forecast-supervised action residual capped by "
            "--action_residual_max_scale."
        ),
    )
    parser.add_argument(
        "--zero_init_action_adapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize f_m action-adapter outputs to zero while retaining a "
            "small bounded scale, so action conditioning starts exactly from "
            "the passive transition but adapter weights learn directly."
        ),
    )
    parser.add_argument(
        "--action_adapter_initial_scale",
        type=float,
        default=0.10,
        help=(
            "Initial bounded residual scale for zero-output action adapters; "
            "must lie in [0, 0.25)."
        ),
    )
    parser.add_argument(
        "--action_residual_max_scale",
        type=float,
        default=0.25,
        help=(
            "Positive maximum magnitude of the bounded action-to-state "
            "residual. The default 0.25 preserves the established protocol."
        ),
    )
    parser.add_argument(
        "--action_adapter_squash",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a bounded inner tanh to the learned f_m action adapters. "
            "This is disabled by default for backward-compatible Action+RL "
            "runs; with zero-initialized adapters, both zero action and the "
            "initial transition remain exactly passive."
        ),
    )
    parser.add_argument(
        "--action_adapter_temperature",
        type=float,
        default=0.25,
        help=(
            "Temperature of the optional bounded action-adapter squash; "
            "must be positive."
        ),
    )
    parser.add_argument(
        "--action_injection_mode",
        choices=["full", "current_residual", "post_norm_residual"],
        default="full",
        help=(
            "How the Controller action enters f_m. 'full' retains action "
            "memory/query/reinforcement plus the current residual; "
            "'current_residual' keeps causal history passive and injects "
            "the action inside the current hidden-state update; "
            "'post_norm_residual' adds that bounded correction after the "
            "unchanged passive update has been normalized."
        ),
    )
    parser.add_argument("--policy_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--latent_normalization",
        choices=["none", "layernorm"],
        default="layernorm",
        help="Normalization applied to online/EMA target node latents.",
    )
    parser.add_argument(
        "--latent_min_logvar",
        type=float,
        default=-6.0,
        help="Lower clamp for the Gaussian future-latent log variance.",
    )
    parser.add_argument(
        "--latent_max_logvar",
        type=float,
        default=2.0,
        help="Upper clamp for the Gaussian future-latent log variance.",
    )
    parser.add_argument(
        "--target_encoder_momentum",
        type=float,
        default=0.99,
        help=(
            "EMA momentum for the stop-gradient future-latent target. "
            "Zero uses the online Graph Module as the detached target."
        ),
    )
    parser.add_argument(
        "--observable_latent_mode",
        choices=["mean", "sample"],
        default="mean",
        help="Decode observables from the Gaussian future-latent mean or sample.",
    )
    parser.add_argument(
        "--property_decoder_zero_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize the T4 property decoder output to zero.  This makes "
            "the initial residual equal to the observed source state."
        ),
    )
    parser.add_argument("--sgt_num_hops", type=int, default=2)
    parser.add_argument("--sgt_num_walks", type=int, default=4)
    parser.add_argument("--sgt_walk_length", type=int, default=3)
    parser.add_argument(
        "--sgt_topology_cache_bytes",
        type=int,
        default=0,
        help=(
            "Bounded cache for deterministic SGT topology/walk tensors. "
            "Stochastic training caches only topology preprocessing."
        ),
    )
    parser.add_argument(
        "--graph_encoder_type",
        choices=["mentor_sgt", "mentor_sgt_gwm"],
        default="mentor_sgt_gwm",
        help=(
            "Graph Module backbone. mentor_sgt matches the established "
            "unweighted passive SGT; mentor_sgt_gwm additionally consumes "
            "observed current edge weights."
        ),
    )
    parser.add_argument(
        "--state_model_type",
        choices=["mentor_history_transformer", "mentor_action_history_transformer", "mentor_sgt_gwm_transformer"],
        default="mentor_sgt_gwm_transformer",
        help=(
            "State Module backbone. mentor_action_history_transformer is "
            "the action-residual extension of the established passive "
            "mentor_history_transformer."
        ),
    )
    parser.add_argument(
        "--sgt_deterministic_walks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use deterministic structural walks. The default keeps the "
            "stochastic random-walk encoder used by the established GWNN "
            "benchmark while remaining reproducible under --seed."
        ),
    )
    parser.add_argument("--history_window", type=int, default=8)
    parser.add_argument("--history_num_heads", type=int, default=4)
    parser.add_argument("--edge_input", choices=["binary", "weighted"], default="binary")
    parser.add_argument(
        "--property_mode",
        choices=[
            "auto",
            "delta",
            "logit_residual",
            "logit_mixture",
            "logit_residual_mixture",
        ],
        default="auto",
        help="Observable property transition; auto uses simplex-preserving logit residual.",
    )
    parser.add_argument(
        "--property_primary_readout",
        choices=["latent_only", "reference_calibrated"],
        default="reference_calibrated",
        help=(
            "T4 observable readout. reference_calibrated uses a validation-only "
            "convex correction of a past-only history reference by D_Y(Zhat); "
            "latent_only reports D_Y(Zhat) directly."
        ),
    )
    parser.add_argument(
        "--property_loss_type",
        choices=["auto", "mse", "smooth_l1", "cross_entropy"],
        default="auto",
        help=(
            "Observable T4 loss. auto uses soft-label cross entropy for "
            "simplex/logit modes and Smooth-L1 for additive delta mode."
        ),
    )
    parser.add_argument(
        "--property_history_input",
        choices=["auto", "on", "off"],
        default="auto",
        help="Append past-only B_t to g_t; auto enables it for Genre/Reddit.",
    )
    parser.add_argument("--lr_world", type=float, default=1e-3)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help=(
            "AdamW weight decay for the World Model. The default matches "
            "the established passive GWNN optimization protocol."
        ),
    )
    parser.add_argument(
        "--forecast_change_pos_weight",
        type=float,
        default=1.0,
        help=(
            "Positive-class multiplier only for the World Model T2/T3 "
            "change BCE. One matches the passive GWNN loss; class balancing "
            "is an explicit extension rather than an implicit action change."
        ),
    )
    parser.add_argument(
        "--forecast_latent_cosine_weight",
        type=float,
        default=0.1,
        help="Cosine regularization coefficient for future-latent forecasting.",
    )
    parser.add_argument(
        "--forecast_latent_variance_weight",
        type=float,
        default=0.001,
        help="High-variance regularization coefficient for Gaussian future latent.",
    )
    parser.add_argument(
        "--world_action_lr_scale",
        type=float,
        default=1.0,
        help=(
            "Learning-rate multiplier for the zero-initialized f_m action "
            "residuals and GraphActionEncoder. Values below one let the "
            "same-seed passive backbone learn at its established rate while "
            "the predicted-action correction enters gradually."
        ),
    )
    parser.add_argument(
        "--action_residual_gate_lr_scale",
        type=float,
        default=1.0,
        help=(
            "Learning-rate multiplier for the bounded scalar action residual "
            "gate. This leaves the passive backbone and action projections "
            "unchanged while controlling how quickly f_m can admit an "
            "action-conditioned correction."
        ),
    )
    parser.add_argument(
        "--freeze_action_residual_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the bounded action-residual gate at its initialized scale. "
            "This is useful for a controlled action-response-only run, where "
            "the action channel must not be suppressed by a competing "
            "observational forecast objective. The standard benchmark keeps "
            "this disabled."
        ),
    )
    parser.add_argument("--lr_controller", type=float, default=1e-3)
    parser.add_argument("--group_beta", type=float, default=1.0)
    parser.add_argument("--group_temperature", type=float, default=1.0)
    parser.add_argument("--grpo_clip", type=float, default=0.2)
    parser.add_argument("--grpo_kl", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument(
        "--node_action_reward_weight",
        type=float,
        default=0.0,
        help=(
            "For T2/T3/T4, mix direct Controller action quality into the "
            "GRPO reward. T3 combines proposed-node localization with the "
            "GWM response magnitude on those nodes; T4 uses the train-fitted "
            "meaningful-property-change threshold. Zero preserves the "
            "decoder-only reward."
        ),
    )
    parser.add_argument(
        "--node_action_value_reward_weight",
        type=float,
        default=0.0,
        help=(
            "Within T3's direct Controller reward, share assigned to the "
            "proposed action value versus the action-conditioned GWM "
            "response. Zero preserves response-only verification."
        ),
    )
    parser.add_argument(
        "--controller_supervised_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only auxiliary BCE for the Controller node-target head. "
            "The next-step labels are never supplied to inference or f_m."
        ),
    )
    parser.add_argument(
        "--controller_state_alignment_weight",
        type=float,
        default=0.0,
        help=(
            "Optional joint state-learning auxiliary for unary T2/T3 tasks. "
            "It backpropagates the Controller's train-time edit supervision "
            "into the current Graph Module representation while the Controller "
            "still receives only G_{<=t} at inference. Zero preserves the "
            "detached-controller protocol."
        ),
    )
    parser.add_argument(
        "--controller_state_alignment_magnitude_weight",
        type=float,
        default=0.0,
        help=(
            "Optional T3 magnitude-NLL share inside the controller-to-current-"
            "state alignment auxiliary. Zero keeps the alignment target to "
            "future-change localization, while the Controller magnitude head "
            "retains its ordinary supervision."
        ),
    )
    parser.add_argument(
        "--action_plan_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional Action+RL auxiliary weight that makes Zhat retain the "
            "Controller's detached, past-only soft action plan."
        ),
    )
    parser.add_argument(
        "--action_plan_positive_weight_cap",
        type=float,
        default=0.0,
        help=(
            "Optional positive-class cap for sparse action-plan consistency "
            "BCE; zero keeps the original unweighted auxiliary."
        ),
    )
    parser.add_argument(
        "--action_decoder_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional node-task auxiliary weight that matches the existing "
            "future-latent change decoder to the Controller's detached, "
            "past-only soft edit plan. Zero preserves the previous protocol."
        ),
    )
    parser.add_argument(
        "--action_decoder_positive_weight_cap",
        type=float,
        default=0.0,
        help=(
            "Optional positive-class cap for sparse direct action-decoder "
            "consistency BCE; zero keeps the original unweighted auxiliary."
        ),
    )
    parser.add_argument(
        "--action_value_decoder_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional T3 auxiliary that matches the future-latent node-state "
            "decoder to the Controller's detached, past-only delta proposal "
            "on its own soft edit support. Zero preserves the prior protocol."
        ),
    )
    parser.add_argument(
        "--action_transition_advantage_weight",
        type=float,
        default=0.0,
        help=(
            "Optional action-effectiveness auxiliary weight: the future "
            "latent under the proposed current-state action must not be "
            "worse than a detached zero-action counterfactual."
        ),
    )
    parser.add_argument(
        "--action_transition_advantage_margin",
        type=float,
        default=0.0,
        help=(
            "Absolute latent-MSE improvement requested over the detached "
            "zero-action counterfactual when action-transition advantage is enabled."
        ),
    )
    parser.add_argument(
        "--action_observable_advantage_weight",
        type=float,
        default=0.0,
        help=(
            "Optional action-effectiveness auxiliary weight: the task-specific "
            "observable loss under the proposed action must not be worse than "
            "an identical detached zero-action counterfactual."
        ),
    )
    parser.add_argument(
        "--action_observable_advantage_margin",
        type=float,
        default=0.0,
        help=(
            "Requested absolute training-observable improvement over the "
            "zero-action counterfactual when observable advantage is enabled."
        ),
    )
    parser.add_argument(
        "--zero_action_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for a same-input zero-action forecast anchor. "
            "The action and zero-action objectives are normalized by one "
            "plus this weight, preserving the overall forecast-loss scale."
        ),
    )
    parser.add_argument(
        "--action_residual_only_world_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train the shared World Model from an identical zero-action "
            "forecast while action-only advantage losses train the bounded "
            "action residual. This is a joint-from-scratch residual protocol, "
            "not passive pretraining."
        ),
    )
    parser.add_argument(
        "--soft_node_action_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For T2, condition f_m on the Controller probability for every "
            "node. Discrete edit sequences are still sampled for GRPO."
        ),
    )
    parser.add_argument(
        "--soft_node_action_mode",
        choices=["expected", "centered", "centered_local", "local"],
        default="centered",
        help=(
            "Node-action representation passed to f_m. centered is the "
            "default because a near-uniform proposal should not masquerade "
            "as node-specific edit evidence; centered_local also removes "
            "the shared action token; local writes only non-negative sparse "
            "edit markers; expected writes the raw Controller probability "
            "at every node."
        ),
    )
    parser.add_argument(
        "--soft_node_action_probability_calibration",
        choices=["none", "past_global_prior", "past_global_prior_mass"],
        default="past_global_prior",
        help=(
            "Convert class-balanced unary Controller logits to the soft "
            "probability written into f_m. past_global_prior estimates the "
            "correction from completed earlier transitions only; "
            "past_global_prior_mass additionally matches the past-only "
            "expected edit mass."
        ),
    )
    parser.add_argument(
        "--soft_node_action_prior_smoothing",
        type=float,
        default=1.0,
        help=(
            "Symmetric pseudo-count used by past_global_prior soft-action "
            "calibration."
        ),
    )
    parser.add_argument(
        "--soft_node_action_topk",
        type=int,
        default=0,
        help=(
            "For unary edit conditioning, retain this many highest-scoring "
            "current-state Controller node proposals. Zero uses the dense "
            "expected-action relaxation."
        ),
    )
    parser.add_argument(
        "--soft_node_action_count_support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For unary state/property actions, use the Controller's "
            "current-state predicted node-edit count to retain a sparse "
            "top proposal support."
        ),
    )
    parser.add_argument(
        "--soft_edge_action_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For T1, condition f_m on the Controller distribution over every "
            "legal addition/deletion pair; pair scores are computed in chunks."
        ),
    )
    parser.add_argument(
        "--soft_edge_action_topk",
        type=int,
        default=0,
        help=(
            "For topology action conditioning, retain this many current-state "
            "Controller pair proposals independently for ADD_EDGE and "
            "REMOVE_EDGE. Zero uses the legacy all-pair categorical softmax."
        ),
    )
    parser.add_argument(
        "--soft_edge_action_count_support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For multi-label topology actions, use the Controller's "
            "current-state predicted ADD/REMOVE counts to retain that many "
            "highest-scoring legal pair proposals. This prevents a dense "
            "all-pair sigmoid relaxation from becoming a uniform node action."
        ),
    )
    parser.add_argument(
        "--soft_edge_action_mode",
        choices=["categorical", "multilabel"],
        default="categorical",
        help=(
            "Categorical represents one mutually-exclusive edge edit; "
            "multilabel represents simultaneous add/remove pair proposals "
            "for snapshot topology evolution."
        ),
    )
    parser.add_argument(
        "--confidence_gated_soft_action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Suppress an uninformative near-uniform soft Controller action; "
            "the gate uses only current policy uncertainty."
        ),
    )
    parser.add_argument(
        "--soft_node_magnitude_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For T3/T4, include the Controller Gaussian mean for every node "
            "in the soft action passed to f_m."
        ),
    )
    parser.add_argument(
        "--change_gated_action_value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject a unary action magnitude only at nodes selected by the "
            "Controller probability. Localization may remain centered, but "
            "the proposed delta is never sign-reversed or globally pooled."
        ),
    )
    parser.add_argument(
        "--action_full_value_conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Encode the concrete full node-state/property action value in "
            "GraphActionEncoder in addition to dimension-independent value "
            "statistics. This preserves which feature/property coordinate "
            "the Controller edits."
        ),
    )
    parser.add_argument(
        "--controller_magnitude_supervised_weight",
        type=float,
        default=0.0,
        help="Weight for supervised Controller action magnitude.",
    )
    parser.add_argument(
        "--controller_magnitude_loss",
        choices=["gaussian_nll", "smooth_l1"],
        default="gaussian_nll",
        help=(
            "Supervision for Controller edit magnitudes. Gaussian NLL also "
            "trains its uncertainty; smooth_l1 trains the mean used by the "
            "world-model action token without rewarding variance collapse."
        ),
    )
    parser.add_argument(
        "--controller_topology_count_weight",
        type=float,
        default=0.0,
        help=(
            "Smooth-L1 supervision weight for the Controller's future-free "
            "log add/remove batch-count prediction in topology tasks."
        ),
    )
    parser.add_argument(
        "--topology_count_target_mode",
        choices=["absolute", "edge_rate"],
        default="absolute",
        help=(
            "Train the topology count head on absolute log counts or on "
            "log counts normalized by the currently observed edge count."
        ),
    )
    parser.add_argument(
        "--topology_count_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition the topology count head on source-time graph-scale "
            "features (edge count, node count, and density)."
        ),
    )
    parser.add_argument(
        "--topology_history_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a causal pair-history prior for topology ranking during "
            "validation and final evaluation."
        ),
    )
    parser.add_argument(
        "--controller_node_count_weight",
        type=float,
        default=0.0,
        help=(
            "Smooth-L1 supervision weight for the Controller's future-free "
            "log node-edit batch-count prediction in unary action tasks."
        ),
    )
    parser.add_argument("--lambda_observable", type=float, default=2.0)
    parser.add_argument("--lambda_action_locality", type=float, default=0.5)
    parser.add_argument("--action_locality_margin", type=float, default=0.25)
    parser.add_argument(
        "--lambda_synthetic_intervention",
        type=float,
        default=1.0,
        help="Weight of the current-state synthetic action-response objective.",
    )
    parser.add_argument(
        "--synthetic_intervention_action_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the synthetic action-response gradient only to the action "
            "encoder and action-residual parameters in f_m. This preserves "
            "the shared observed-forecast backbone while teaching the "
            "action-specific pathway an external-edit response."
        ),
    )
    parser.add_argument(
        "--synthetic_action_delta_scale",
        type=float,
        default=0.10,
        help=(
            "Maximum absolute physical unary edit used by the source-only "
            "synthetic action-response target; policy Gaussian values are "
            "mapped through scale*tanh before f_m and Apply(g_t, a_t)."
        ),
    )
    parser.add_argument(
        "--joint_action_groups",
        "--synthetic_topology_actions",
        dest="joint_action_groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in joint-group protocol. T1 adds future-free synthetic "
            "MODIFY_NODE_STATE to its native ADD/REMOVE groups; T2/T3/T4 "
            "retain their native MODIFY group and additionally train "
            "synthetic ADD_EDGE and REMOVE_EDGE groups. The legacy "
            "--synthetic_topology_actions spelling is an alias."
        ),
    )
    parser.add_argument(
        "--joint_action_group_interval",
        "--synthetic_topology_interval",
        dest="joint_action_group_interval",
        type=int,
        default=1,
        help=(
            "Apply the optional joint ADD/REMOVE intervention "
            "augmentation once every this many chronological training "
            "steps. Real one-step forecasting is still optimized at every "
            "step. The legacy --synthetic_topology_interval spelling is an "
            "alias."
        ),
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=1.0,
        help="Weight of supervised G_t,a_t -> observed G_(t+1) forecasting.",
    )
    parser.add_argument("--forecast_latent_weight", type=float, default=1.0)
    parser.add_argument("--forecast_change_weight", type=float, default=1.0)
    parser.add_argument("--forecast_state_weight", type=float, default=1.0)
    parser.add_argument(
        "--topology_count_weight",
        type=float,
        default=0.0,
        help=(
            "Smooth-L1 weight for predicting log(1+addition/removal count) "
            "from the future latent. Zero preserves the former protocol."
        ),
    )
    parser.add_argument(
        "--forecast_state_loss_mode",
        choices=["all", "changed_only", "conditional_changed"],
        default="all",
    )
    parser.add_argument(
        "--forecast_state_loss_type",
        choices=["smooth_l1", "l1"],
        default="smooth_l1",
        help="Regression loss for the observed T3 state transition target.",
    )
    parser.add_argument(
        "--node_state_selection_metric",
        choices=["mae", "rmse"],
        default="rmse",
        help="Validation metric used to select the T3 checkpoint.",
    )
    parser.add_argument(
        "--node_change_selection_metric",
        choices=["auprc", "f1", "harmonic"],
        default="auprc",
        help=(
            "Validation metric used to select the T2 checkpoint. 'harmonic' "
            "balances calibrated AUPRC and F1 without accessing test data."
        ),
    )
    parser.add_argument("--forecast_all_state_weight", type=float, default=0.25)
    parser.add_argument(
        "--forecast_current_state_weight",
        type=float,
        default=0.0,
        help="Past-only current-state reconstruction regularizer for T3.",
    )
    parser.add_argument(
        "--state_prediction_mode",
        choices=["ungated", "gated"],
        default="ungated",
    )
    parser.add_argument(
        "--node_change_decoder_input",
        choices=[
            "future",
            "latent_delta",
            "latent_abs_delta",
            "future_concat_delta",
            "future_concat_abs_delta",
        ],
        default="future_concat_abs_delta",
        help=(
            "Latent readout supplied to the auxiliary node-change decoder. "
            "T3 may use 'future' to match its passive state-regression "
            "protocol exactly."
        ),
    )
    parser.add_argument(
        "--state_decoder_zero_init",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--action_conditioning_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Keep f_m on the exact passive path for this many initial epochs "
            "while the Controller learns future-free proposals. The Controller "
            "still receives its supervised/RL updates; only its untrained action "
            "is withheld from the recurrent world state."
        ),
    )
    parser.add_argument(
        "--select_passive_warmup_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Permit validation to select a warm-up checkpoint while the "
            "bounded action residual is exactly zero. This preserves a "
            "validation-selected passive fallback if Controller actions are "
            "not yet useful."
        ),
    )
    parser.add_argument("--world_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--controller_warmup_epochs",
        type=int,
        default=None,
        help=(
            "Number of epochs before GRPO Controller updates begin. When "
            "omitted, preserves the legacy coupling to --world_warmup_epochs; "
            "set to zero to pretrain the future-free Controller while the "
            "action residual in f_m is still warming up."
        ),
    )
    parser.add_argument(
        "--topology_exact_reward_weight",
        type=float,
        default=0.5,
        help="Weight of exact set F1; the remainder is dense balanced shaping.",
    )
    parser.add_argument(
        "--topology_action_reward_weight",
        type=float,
        default=0.0,
        help=(
            "Share of topology GRPO reward assigned to precision of the "
            "Controller's proposed ADD/REMOVE pairs. Zero preserves the "
            "decoder-only topology reward."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/action_rl_smoke.pt")
    parser.add_argument(
        "--pretrained_backbone",
        default=None,
        help=(
            "Optional LOO-pretrained WorldGraph graph/state/latent backbone. "
            "Task-specific Action, Controller and decoder modules remain fresh."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_mode",
        choices=["structural", "legacy"],
        default="structural",
        help=(
            "structural keeps the target raw-feature input and transfers only "
            "feature-independent SGT weights; legacy uses the fixed-width adapter."
        ),
    )
    parser.add_argument(
        "--pretrained_backbone_blend",
        type=float,
        default=0.25,
        help="Fraction of compatible pretrained weights to load in structural mode.",
    )
    parser.add_argument(
        "--pretrained_graph_encoder_blend",
        type=float,
        default=None,
        help="Optional graph-encoder-specific pretrained blend fraction.",
    )
    parser.add_argument(
        "--pretrained_state_model_blend",
        type=float,
        default=None,
        help="Optional history-state-specific pretrained blend fraction.",
    )
    parser.add_argument(
        "--pretrained_latent_predictor_blend",
        type=float,
        default=None,
        help="Optional latent-predictor-specific pretrained blend fraction.",
    )
    parser.add_argument(
        "--pretrained_task_blend",
        type=float,
        default=0.5,
        help="Blend for the optional task-aligned edge-transition module.",
    )
    parser.add_argument(
        "--pretrained_transfer_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For T2 topology transfer, keep a fresh target-domain backbone and "
            "fuse a frozen LOO graph branch through learnable conservative gates. "
            "The source Controller is not loaded in this mode."
        ),
    )
    parser.add_argument(
        "--pretrained_transfer_gate_initial",
        type=float,
        default=0.1,
        help="Initial contribution of the frozen T2 graph-transfer branch.",
    )
    parser.add_argument(
        "--pretrained_transfer_state_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Transfer action-independent history-state layers. Structural mode "
            "always skips source action projections and gates."
        ),
    )
    parser.add_argument("--results", default="results/action_rl_smoke.json")
    parser.add_argument(
        "--validation_transitions",
        type=int,
        default=0,
        help="Maximum target-split transitions used for checkpoint validation; 0 uses all.",
    )
    parser.add_argument(
        "--validation_history_burn_in",
        type=int,
        default=0,
        help=(
            "Number of immediately preceding snapshots used to reconstruct "
            "learned hidden state before validation; observable past-only "
            "statistics are still accumulated over the complete prefix."
        ),
    )
    parser.add_argument(
        "--validation_device",
        default=None,
        help="Optional validation subprocess device (for example cpu for memory-heavy data).",
    )
    parser.add_argument(
        "--inprocess_validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally run repeated checkpoint validation in the training "
            "process while preserving its RNG state; disabled by default."
        ),
    )
    parser.add_argument(
        "--persistent_validation_worker",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse one external evaluator process across epochs. This "
            "preserves process isolation while amortizing interpreter and "
            "CUDA startup; enabled by default for topology/T2."
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print one compact training record every N updates and at each epoch end.",
    )
    args = parser.parse_args()
    if args.inprocess_validation and args.persistent_validation_worker is True:
        raise ValueError(
            "--inprocess_validation and --persistent_validation_worker are "
            "mutually exclusive."
        )
    if args.persistent_validation_worker is None:
        args.persistent_validation_worker = (
            args.task == "topology" and not args.inprocess_validation
        )
    if args.batch_topology_grpo_rollouts is None:
        args.batch_topology_grpo_rollouts = args.task == "topology"
    if args.batch_topology_grpo_rewards is None:
        args.batch_topology_grpo_rewards = args.task == "topology"
    if args.dataset is not None and args.dataset not in ACTION_BENCHMARK_DATASETS[args.task]:
        parser.error(
            f"--task {args.task} supports --dataset: "
            + ", ".join(ACTION_BENCHMARK_DATASETS[args.task])
        )
    if args.steps < 0 or args.epochs < 1:
        parser.error("--steps must be non-negative and --epochs must be positive.")
    if (
        args.validation_transitions < 0
        or args.validation_history_burn_in < 0
        or args.clip_history_burn_in < 0
    ):
        parser.error("Transition and history burn-in budgets must be non-negative.")
    if args.log_every < 1:
        parser.error("--log_every must be positive.")
    if args.val_every < 0 or args.patience < 0:
        parser.error("--val_every and --patience must be non-negative.")
    if args.action_dim < 1:
        parser.error("--action_dim must be positive.")
    if args.grpo_update_interval < 1:
        parser.error("--grpo_update_interval must be positive.")
    if min(
        args.latent_dim,
        args.hidden_dim,
        args.sgt_num_hops,
        args.sgt_num_walks,
        args.sgt_walk_length,
        args.history_window,
        args.history_num_heads,
    ) < 1:
        parser.error("Model dimensions and SGT/history sizes must be positive.")
    if args.hidden_dim % args.history_num_heads:
        parser.error("--hidden_dim must be divisible by --history_num_heads.")
    if args.min_actions < 1 or args.max_actions < args.min_actions:
        parser.error("Require 1 <= --min_actions <= --max_actions.")
    if not 0.0 <= args.inference_stop_threshold <= 1.0:
        parser.error("--inference_stop_threshold must lie in [0, 1].")
    if args.world_warmup_epochs < 0:
        parser.error("--world_warmup_epochs must be non-negative.")
    if (
        args.controller_warmup_epochs is not None
        and args.controller_warmup_epochs < 0
    ):
        parser.error("--controller_warmup_epochs must be non-negative when set.")
    if args.action_conditioning_warmup_epochs < 0:
        parser.error("--action_conditioning_warmup_epochs must be non-negative.")
    if args.world_action_lr_scale <= 0.0:
        parser.error("--world_action_lr_scale must be positive.")
    if args.action_residual_gate_lr_scale <= 0.0:
        parser.error("--action_residual_gate_lr_scale must be positive.")
    if args.action_residual_max_scale <= 0.0:
        parser.error("--action_residual_max_scale must be positive.")
    adapter_scale_cap = (
        args.action_residual_max_scale if args.bounded_action_residual else 0.25
    )
    if not 0.0 <= args.action_adapter_initial_scale < adapter_scale_cap:
        parser.error(
            "--action_adapter_initial_scale must lie in [0, the active residual cap)."
        )
    if args.action_adapter_temperature <= 0.0:
        parser.error("--action_adapter_temperature must be positive.")
    if args.synthetic_action_delta_scale <= 0.0:
        parser.error("--synthetic_action_delta_scale must be positive.")
    if args.weight_decay < 0.0:
        parser.error("--weight_decay must be non-negative.")
    if args.forecast_change_pos_weight <= 0.0:
        parser.error("--forecast_change_pos_weight must be positive.")
    if not 0.0 <= args.target_encoder_momentum < 1.0:
        parser.error("--target_encoder_momentum must lie in [0, 1).")
    if args.latent_min_logvar >= args.latent_max_logvar:
        parser.error("--latent_min_logvar must be smaller than --latent_max_logvar.")
    if (
        args.forecast_latent_cosine_weight < 0.0
        or args.forecast_latent_variance_weight < 0.0
    ):
        parser.error("Future-latent auxiliary weights must be non-negative.")
    if args.soft_edge_action_topk < 0:
        parser.error("--soft_edge_action_topk must be non-negative.")
    if args.soft_node_action_topk < 0:
        parser.error("--soft_node_action_topk must be non-negative.")
    if args.soft_node_action_prior_smoothing <= 0.0:
        parser.error("--soft_node_action_prior_smoothing must be positive.")
    if args.topology_addition_neg_ratio < 0.0:
        parser.error("--topology_addition_neg_ratio must be non-negative.")
    if not 0.0 <= args.topology_exact_reward_weight <= 1.0:
        parser.error("--topology_exact_reward_weight must be in [0, 1].")
    if not 0.0 <= args.topology_action_reward_weight <= 1.0:
        parser.error("--topology_action_reward_weight must be in [0, 1].")
    if not 0.0 <= args.node_action_reward_weight <= 1.0:
        parser.error("--node_action_reward_weight must be in [0, 1].")
    if not 0.0 <= args.node_action_value_reward_weight <= 1.0:
        parser.error("--node_action_value_reward_weight must be in [0, 1].")
    if args.controller_supervised_weight < 0.0:
        parser.error("--controller_supervised_weight must be non-negative.")
    if args.controller_state_alignment_weight < 0.0:
        parser.error("--controller_state_alignment_weight must be non-negative.")
    if args.controller_state_alignment_magnitude_weight < 0.0:
        parser.error(
            "--controller_state_alignment_magnitude_weight must be non-negative."
        )
    if args.action_plan_consistency_weight < 0.0:
        parser.error("--action_plan_consistency_weight must be non-negative.")
    if args.action_plan_positive_weight_cap < 0.0:
        parser.error("--action_plan_positive_weight_cap must be non-negative.")
    if args.action_decoder_consistency_weight < 0.0:
        parser.error(
            "--action_decoder_consistency_weight must be non-negative."
        )
    if args.action_decoder_positive_weight_cap < 0.0:
        parser.error(
            "--action_decoder_positive_weight_cap must be non-negative."
        )
    if args.action_value_decoder_consistency_weight < 0.0:
        parser.error(
            "--action_value_decoder_consistency_weight must be non-negative."
        )
    if (
        args.action_value_decoder_consistency_weight > 0.0
        and args.task != "node_state"
    ):
        parser.error(
            "--action_value_decoder_consistency_weight is currently defined "
            "only for node-state transitions."
        )
    if args.action_transition_advantage_weight < 0.0:
        parser.error(
            "--action_transition_advantage_weight must be non-negative."
        )
    if args.action_transition_advantage_margin < 0.0:
        parser.error(
            "--action_transition_advantage_margin must be non-negative."
        )
    if args.action_observable_advantage_weight < 0.0:
        parser.error(
            "--action_observable_advantage_weight must be non-negative."
        )
    if args.action_observable_advantage_margin < 0.0:
        parser.error(
            "--action_observable_advantage_margin must be non-negative."
        )
    if args.zero_action_anchor_weight < 0.0:
        parser.error("--zero_action_anchor_weight must be non-negative.")
    if (
        args.action_residual_only_world_loss
        and args.action_transition_advantage_weight <= 0.0
        and args.action_observable_advantage_weight <= 0.0
        and args.lambda_synthetic_intervention <= 0.0
    ):
        parser.error(
            "--action_residual_only_world_loss requires an action-specific "
            "objective: enable a transition/observable advantage or synthetic "
            "intervention loss."
        )
    if (
        args.action_residual_only_world_loss
        and args.zero_init_action_adapters
        and args.lambda_synthetic_intervention <= 0.0
        and not (
            (
                args.action_transition_advantage_weight > 0.0
                and args.action_transition_advantage_margin > 0.0
            )
            or (
                args.action_observable_advantage_weight > 0.0
                and args.action_observable_advantage_margin > 0.0
            )
        )
    ):
        parser.error(
            "With --action_residual_only_world_loss and zero-initialized "
            "action adapters, use a positive transition/observable advantage "
            "margin or a synthetic intervention loss. Otherwise the action "
            "and zero-action branches coincide initially and the action "
            "adapter receives no learning signal."
        )
    if (
        args.action_decoder_consistency_weight > 0.0
        and args.task == "topology"
    ):
        parser.error(
            "--action_decoder_consistency_weight is currently defined only "
            "for node-edit tasks."
        )
    if args.controller_magnitude_supervised_weight < 0.0:
        parser.error("--controller_magnitude_supervised_weight must be non-negative.")
    if args.controller_topology_count_weight < 0.0:
        parser.error("--controller_topology_count_weight must be non-negative.")
    if args.controller_node_count_weight < 0.0:
        parser.error("--controller_node_count_weight must be non-negative.")
    if args.lambda_action_locality < 0.0 or args.action_locality_margin < 0.0:
        parser.error("Action-locality weight and margin must be non-negative.")
    if (
        args.joint_action_groups
        and args.lambda_synthetic_intervention <= 0.0
    ):
        parser.error(
            "--joint_action_groups requires a positive "
            "--lambda_synthetic_intervention."
        )
    if args.joint_action_group_interval < 1:
        parser.error("--joint_action_group_interval must be positive.")
    nonnegative_forecast = (
        args.lambda_synthetic_intervention,
        args.lambda_forecast,
        args.forecast_latent_weight,
        args.forecast_change_weight,
        args.forecast_state_weight,
        args.forecast_all_state_weight,
        args.forecast_current_state_weight,
        args.topology_count_weight,
    )
    if any(value < 0.0 for value in nonnegative_forecast):
        parser.error("Forecast and synthetic objective weights must be non-negative.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(
        "runtime_cpu="
        f"intraop_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )
    dataset_name = args.dataset or ACTION_BENCHMARK_DATASETS[args.task][0]
    processed = resolve_action_benchmark_path(
        ROOT, args.task, dataset_name, args.processed
    )
    if args.processed is not None and args.dataset is None:
        dataset_name = infer_action_benchmark_dataset(args.task, processed)
    metadata, transitions = _graph_transitions(
        processed,
        args.task,
        dataset=dataset_name,
        edge_input=args.edge_input,
    )
    property_mode = (
        "logit_residual" if args.property_mode == "auto" else args.property_mode
    )
    property_history_input = args.task == "node_property" and (
        args.property_history_input == "on"
        or (args.property_history_input == "auto" and dataset_name in {"genre", "reddit"})
    )
    if not transitions:
        raise RuntimeError("No chronological training transitions are available.")
    num_nodes = int(metadata["num_nodes"])
    node_feature_dim = int(transitions[0]["x_t"].shape[1])
    property_dim = (
        int(metadata["official_target_dim"]) if args.task == "node_property" else None
    )
    property_change_threshold = (
        _train_property_change_threshold(transitions)
        if args.task == "node_property"
        and (
            args.controller_supervised_weight > 0.0
            or args.forecast_change_weight > 0.0
        )
        else None
    )
    input_dim = node_feature_dim + (
        (property_dim or 0) if property_history_input else 0
    )

    model = GraphWorldModel(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        node_property_dim=property_dim,
        property_decoder_zero_init=(
            args.property_decoder_zero_init if args.task == "node_property" else False
        ),
        state_decoder_zero_init=args.state_decoder_zero_init,
        state_prediction_mode=args.state_prediction_mode,
        current_state_reconstruction=(
            args.task in {"node_change", "node_state"}
            and args.forecast_current_state_weight > 0.0
        ),
        topology_transition_heads=(
            args.task == "topology" or args.joint_action_groups
        ),
        topology_count_decoder=(
            args.task == "topology" and args.topology_count_weight > 0.0
        ),
        topology_pair_state_dim=(
            PAIR_STATE_DIM
            if args.task == "topology"
            and (
                args.topology_pair_state
                or args.topology_candidate_policy == "seen_pairs"
            )
            else 0
        ),
        dropout=args.dropout,
        observable_latent_mode=args.observable_latent_mode,
        target_encoder_momentum=args.target_encoder_momentum,
        latent_normalization=args.latent_normalization,
        latent_min_logvar=args.latent_min_logvar,
        latent_max_logvar=args.latent_max_logvar,
        graph_encoder_type=args.graph_encoder_type,
        state_model_type=args.state_model_type,
        sgt_num_hops=args.sgt_num_hops,
        sgt_num_walks=args.sgt_num_walks,
        sgt_walk_length=args.sgt_walk_length,
        sgt_topology_cache_bytes=args.sgt_topology_cache_bytes,
        sgt_deterministic_walks=args.sgt_deterministic_walks,
        history_window=args.history_window,
        history_num_heads=args.history_num_heads,
        separate_action_query=args.separate_action_query,
        bounded_action_residual=args.bounded_action_residual,
        zero_init_action_adapters=args.zero_init_action_adapters,
        action_adapter_initial_scale=args.action_adapter_initial_scale,
        action_residual_max_scale=args.action_residual_max_scale,
        action_adapter_squash=args.action_adapter_squash,
        action_adapter_temperature=args.action_adapter_temperature,
        action_injection_mode=args.action_injection_mode,


        input_adapter_type=(
            "residual_domain_invariant"
            if args.pretrained_backbone and args.pretrained_transfer_mode == "legacy"
            else "none"
        ),
        action_plan_decoder=args.action_plan_consistency_weight > 0.0,
        action_plan_decoder_dim=(
            2
            if args.task == "topology" and args.action_plan_consistency_weight > 0.0
            else 1
        ),
        node_change_decoder_input=args.node_change_decoder_input,
    ).to(device)
    if args.pretrained_transfer_gate:
        if args.task != "topology":
            raise ValueError("T2 pretrained transfer gates require task=topology.")
        if not args.pretrained_backbone:
            raise ValueError("--pretrained_transfer_gate requires --pretrained_backbone.")
        if not 0.0 < args.pretrained_transfer_gate_initial < 1.0:
            raise ValueError("--pretrained_transfer_gate_initial must lie in (0, 1).")
        source_shell = GraphWorldModel(
            input_dim=args.latent_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            dropout=args.dropout,
            observable_latent_mode=args.observable_latent_mode,
            latent_normalization=args.latent_normalization,
            graph_encoder_type=args.graph_encoder_type,
            state_model_type=args.state_model_type,
            sgt_num_hops=args.sgt_num_hops,
            sgt_num_walks=args.sgt_num_walks,
            sgt_walk_length=args.sgt_walk_length,
            sgt_topology_cache_bytes=max(0, args.sgt_topology_cache_bytes // 2),
            sgt_deterministic_walks=args.sgt_deterministic_walks,
            history_window=args.history_window,
            history_num_heads=args.history_num_heads,
            separate_action_query=args.separate_action_query,
            bounded_action_residual=args.bounded_action_residual,
            zero_init_action_adapters=args.zero_init_action_adapters,
            action_adapter_initial_scale=args.action_adapter_initial_scale,
            action_residual_max_scale=args.action_residual_max_scale,
            action_adapter_squash=args.action_adapter_squash,
            action_adapter_temperature=args.action_adapter_temperature,
            action_injection_mode=args.action_injection_mode,
            input_adapter_type="domain_invariant",
        )
        transfer_summary = load_worldgraph_pretrained_backbone(
            source_shell,
            args.pretrained_backbone,
            map_location="cpu",
            expected_task="T2",
            expected_dataset=dataset_name,
            transfer_mode="legacy",
            transfer_state_model=False,
            backbone_blend=1.0,
        )
        assert source_shell.input_adapter is not None
        model.pretrained_graph_transfer_branch = FrozenGraphTransferBranch(
            source_shell.input_adapter,
            source_shell.graph_encoder,
            args.latent_dim,
            initial_gate=args.pretrained_transfer_gate_initial,
        ).to(device)
        model.topology_add_transfer_adapter = GatedResidualTransferAdapter(
            args.latent_dim,
            bottleneck=max(4, args.latent_dim // 2),
            initial_gate=args.pretrained_transfer_gate_initial,
        ).to(device)
        model.topology_remove_transfer_adapter = GatedResidualTransferAdapter(
            args.latent_dim,
            bottleneck=max(4, args.latent_dim // 2),
            initial_gate=args.pretrained_transfer_gate_initial,
        ).to(device)
        print(
            "loaded_frozen_pretrained_graph_branch="
            + str(transfer_summary["path"])
            + " tensors="
            + str(transfer_summary["loaded_tensors"])
            + " source_controller_loaded=false",
            flush=True,
        )
    elif args.pretrained_backbone:
        summary = load_worldgraph_pretrained_backbone(
            model,
            args.pretrained_backbone,
            map_location="cpu",
            expected_task="T2",
            expected_dataset=dataset_name,
            transfer_mode=args.pretrained_transfer_mode,
            transfer_state_model=args.pretrained_transfer_state_model,
            backbone_blend=args.pretrained_backbone_blend,
            graph_encoder_blend=args.pretrained_graph_encoder_blend,
            state_model_blend=args.pretrained_state_model_blend,
            latent_predictor_blend=args.pretrained_latent_predictor_blend,
        )
        print(
            "loaded_pretrained_backbone="
            + str(summary["path"])
            + " tensors="
            + str(summary["loaded_tensors"])
            + " by_component="
            + json.dumps(summary["loaded_by_component"], sort_keys=True),
            flush=True,
        )







    with torch.random.fork_rng(devices=[]):
        action_encoder = GraphActionEncoder(
            args.latent_dim,
            args.action_dim,
            node_state_dim=(
                node_feature_dim
                if args.task == "node_state"
                and (
                    args.action_full_value_conditioning
                    or args.soft_node_magnitude_conditioning
                )
                else None
            ),
            node_property_dim=(
                property_dim
                if args.task == "node_property"
                and (
                    args.action_full_value_conditioning
                    or args.soft_node_magnitude_conditioning
                )
                else None
            ),
        ).to(device)
        controller = ActionAwareController(
            args.latent_dim,
            args.hidden_dim,
            policy_dim=args.policy_dim,
            node_state_dim=node_feature_dim,
            node_property_dim=property_dim,
            topology_count_context_dim=(4 if args.topology_count_context else 0),
        ).to(device)
        if args.pretrained_backbone and not args.pretrained_transfer_gate:
            task_summary = load_worldgraph_pretrained_task_module(
                controller,
                args.pretrained_backbone,
                source_prefix="topology_controller",
                blend=args.pretrained_task_blend,
            )
            if task_summary is not None:
                print(
                    "loaded_pretrained_task_topology_controller="
                    + json.dumps(task_summary, sort_keys=True),
                    flush=True,
                )
    controller_parameters = tuple(
        parameter for parameter in controller.parameters() if parameter.requires_grad
    )
    action_residual_gate_ids = {
        id(parameter)
        for name, parameter in model.state_model.named_parameters()
        if name == "action_residual_gate"
    }
    action_parameter_ids = {
        id(parameter) for parameter in action_encoder.parameters()
    }
    action_parameter_ids.update(
        id(parameter)
        for name, parameter in model.state_model.named_parameters()
        if name.startswith("action_") and name != "action_residual_gate"
    )
    base_world_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in action_parameter_ids
        and id(parameter) not in action_residual_gate_ids
        and parameter.requires_grad
    ]
    action_world_parameters = [
        parameter
        for parameter in [*model.parameters(), *action_encoder.parameters()]
        if id(parameter) in action_parameter_ids
        and parameter.requires_grad
    ]
    action_residual_gate_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in action_residual_gate_ids
    ]
    if args.freeze_action_residual_gate:
        for parameter in action_residual_gate_parameters:
            parameter.requires_grad_(False)





    action_advantage_parameters = tuple(
        [
            *action_world_parameters,
            *[
                parameter
                for parameter in action_residual_gate_parameters
                if parameter.requires_grad
            ],
        ]
    )






    action_response_parameters = action_advantage_parameters
    world_parameter_groups: list[dict[str, object]] = [
        {"params": base_world_parameters, "lr": args.lr_world},
        {
            "params": action_world_parameters,
            "lr": args.lr_world * args.world_action_lr_scale,
        },
    ]
    if action_residual_gate_parameters and not args.freeze_action_residual_gate:



        world_parameter_groups.append(
            {
                "params": action_residual_gate_parameters,
                "lr": args.lr_world * args.action_residual_gate_lr_scale,
                "weight_decay": 0.0,
            }
        )
    world_optimizer = torch.optim.AdamW(
        world_parameter_groups,
        lr=args.lr_world,
        weight_decay=args.weight_decay,
    )
    controller_optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.lr_controller
    )
    group_sampler = DynamicGroupSampler(
        beta=args.group_beta, temperature=args.group_temperature
    )
    executed_operations_by_transition: dict[
        int, tuple[GraphOperation, ...]
    ] = {}
    observed_operations_by_transition: dict[
        int, tuple[GraphOperation, ...]
    ] = {}
    observed_operations_prefix_index = 0
    hidden = model.initial_hidden(num_nodes, device)
    historical_change_counts = torch.zeros(num_nodes, device=device)
    historical_change_observations = torch.zeros((), device=device)
    historical_change_prefix_counts = torch.zeros(num_nodes, device=device)
    historical_change_prefix_observations = torch.zeros((), device=device)
    historical_change_prefix_index = 0
    topology_pair_state: CausalPairState | None = None
    property_graph_history: GenreObservedPropertyHistory | None = None
    property_prefix_history: GenreObservedPropertyHistory | None = None
    property_prefix_index = 0
    property_clip_prefix_history: GenreObservedPropertyHistory | None = None
    property_clip_prefix_index = 0
    if property_history_input:
        coordinate_nodes = metadata.get(
            "num_genre_coordinate_nodes",
            metadata.get("num_target_coordinate_nodes"),
        )
        if coordinate_nodes is None:
            raise ValueError(
                "Property-history graph state requires a coordinate-node count."
            )
        property_graph_history = GenreObservedPropertyHistory(
            num_nodes,
            int(property_dim),
            num_genre_coordinate_nodes=int(coordinate_nodes),
        )
        property_prefix_history = property_graph_history.copy()




        property_clip_prefix_history = property_graph_history.copy()
    generator = torch.Generator().manual_seed(args.seed)
    logs: list[dict[str, Any]] = []
    checkpoint_path = ROOT / args.checkpoint
    result_path = ROOT / args.results
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.last{checkpoint_path.suffix}"
    )
    validation_result_path = result_path.with_name(
        f"{result_path.stem}.validation{result_path.suffix}"
    )
    validation_history: list[dict[str, Any]] = []
    best_validation_score: float | None = None
    best_epoch: int | None = None
    stopped_early = False
    profile_timing = os.environ.get("GWM_PROFILE_TIMING", "0") == "1"
    validation_worker: _PersistentValidationWorker | None = None
    deterministic_topology_candidate_cache: dict[
        int, dict[GraphOperation, torch.Tensor]
    ] | None = (
        {} if args.cache_deterministic_topology_candidates else None
    )
    device_topology_candidate_cache: dict[
        int, dict[GraphOperation, torch.Tensor]
    ] | None = (
        {} if args.cache_device_topology_candidates else None
    )
    device_topology_pair_context_cache: dict[
        int, dict[GraphOperation, torch.Tensor | None]
    ] | None = (
        {} if args.cache_device_topology_candidates else None
    )

    def profile_mark() -> float:

        if not profile_timing:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def checkpoint_payload(epoch_index: int) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "action_encoder_state": action_encoder.state_dict(),
            "controller_state": controller.state_dict(),
            "group_sampler_state": group_sampler.state_dict(),
            "args": vars(args),
            "metadata": metadata,
            "property_change_threshold_train_q75": property_change_threshold,
            "saved_epoch": int(epoch_index + 1),
        }

    def epoch_transition_indices(epoch_index: int) -> range:

        return _rotating_clip_indices(
            len(transitions), args.steps, epoch_index, args.epochs
        )



    training_stream = (
        (epoch, epoch_position, transition_index, transitions[transition_index])
        for epoch in range(args.epochs)
        for epoch_position, transition_index in enumerate(
            epoch_transition_indices(epoch)
        )
    )
    for global_step, (
        epoch,
        epoch_position,
        transition_index,
        transition_cpu,
    ) in enumerate(training_stream):
        profile_start = profile_mark()





        world_action_enabled = bool(
            epoch >= args.action_conditioning_warmup_epochs
        )
        if epoch_position == 0:
            hidden = model.initial_hidden(num_nodes, device)




            replay_unary_controller_actions = bool(
                args.clip_history_action_replay
                and world_action_enabled
                and args.task in {"node_change", "node_state"}
                and args.clip_history_burn_in > 0
                and transition_index > 0
            )





            if not replay_unary_controller_actions:
                hidden = _warm_start_clip_hidden_from_observed_prefix(
                    model,
                    transitions,
                    hidden,
                    transition_index=transition_index,
                    burn_in=args.clip_history_burn_in,
                    device=device,
                    task=args.task,
                    edge_input=args.edge_input,
                    property_history_input=property_history_input,
                )
            if (
                not replay_unary_controller_actions
                and
                args.task != "node_property"
                and args.clip_history_burn_in > 0
                and transition_index > 0
            ):
                clip_begin = max(
                    0, transition_index - args.clip_history_burn_in
                )
                print(
                    "clip_history_warm_start "
                    f"source_transitions={clip_begin}:{transition_index} "
                    "future_targets=0"
                )



            while observed_operations_prefix_index < transition_index:
                observed_operations_by_transition[
                    observed_operations_prefix_index
                ] = _observed_operation_groups(
                    args.task,
                    transitions[observed_operations_prefix_index],
                )
                observed_operations_prefix_index += 1
            _restore_group_sampler_prefix(
                group_sampler,
                executed_operations_by_transition,
                transition_index,
                fallback_operations_by_transition=observed_operations_by_transition,
            )
            while historical_change_prefix_index < transition_index:
                prefix_transition = transitions[historical_change_prefix_index]
                historical_change_prefix_counts += _past_change_update(
                    args.task,
                    prefix_transition,
                    device,
                    num_nodes=num_nodes,
                )
                historical_change_prefix_observations += (
                    _past_change_observation_count(
                        args.task, prefix_transition, device
                    )
                )
                historical_change_prefix_index += 1
            historical_change_counts = historical_change_prefix_counts.clone()
            historical_change_observations = (
                historical_change_prefix_observations.clone()
            )
            if replay_unary_controller_actions:
                hidden = _warm_start_unary_action_history_from_observed_prefix(
                    model,
                    action_encoder,
                    controller,
                    transitions,
                    hidden,
                    transition_index=transition_index,
                    burn_in=args.clip_history_burn_in,
                    device=device,
                    task=args.task,
                    edge_input=args.edge_input,
                    soft_node_action_mode=args.soft_node_action_mode,
                    soft_node_action_topk=args.soft_node_action_topk,
                    soft_node_action_probability_calibration=(
                        args.soft_node_action_probability_calibration
                    ),
                    soft_node_action_prior_smoothing=(
                        args.soft_node_action_prior_smoothing
                    ),
                    confidence_gated_soft_action=(
                        args.confidence_gated_soft_action
                    ),
                    soft_node_magnitude_conditioning=(
                        args.soft_node_magnitude_conditioning
                    ),
                    change_gated_action_value=args.change_gated_action_value,
                )
                clip_begin = max(
                    0, transition_index - args.clip_history_burn_in
                )
                print(
                    "clip_history_warm_start "
                    f"source_transitions={clip_begin}:{transition_index} "
                    "action_history=proposal future_targets=0"
                )
            if args.task == "topology" and (
                args.topology_pair_state
                or args.topology_candidate_policy == "seen_pairs"
            ):
                topology_pair_state = CausalPairState(num_nodes)




                for prefix_index in range(transition_index):
                    topology_pair_state.observe(
                        transitions[prefix_index]["edge_index_t"],
                        time_index=prefix_index,
                    )
            if property_history_input:





                assert property_prefix_history is not None
                while property_prefix_index < transition_index:
                    prior = transitions[property_prefix_index]
                    property_prefix_history.observe(
                        prior["property_node_ids_t"],
                        prior["property_observed_t"],
                    )
                    property_prefix_index += 1
                property_graph_history = property_prefix_history.copy()
                if args.clip_history_burn_in and transition_index:
                    assert property_clip_prefix_history is not None
                    clip_begin = max(
                        0, transition_index - args.clip_history_burn_in
                    )
                    while property_clip_prefix_index < clip_begin:
                        prior = transitions[property_clip_prefix_index]
                        property_clip_prefix_history.observe(
                            prior["property_node_ids_t"],
                            prior["property_observed_t"],
                        )
                        property_clip_prefix_index += 1
                    hidden = _warm_start_property_clip_hidden_from_observed_prefix(
                        model,
                        transitions,
                        hidden,
                        property_clip_prefix_history,
                        transition_index=transition_index,
                        burn_in=args.clip_history_burn_in,
                        device=device,
                        edge_input=args.edge_input,
                    )
                    print(
                        "clip_history_warm_start "
                        f"source_transitions={clip_begin}:{transition_index} "
                        "property_history=observed_only future_targets=0"
                    )
        transition = _transition_to_device(
            transition_cpu, device, task=args.task
        )
        if property_graph_history is not None:
            property_graph_history.observe(
                transition_cpu["property_node_ids_t"],
                transition_cpu["property_observed_t"],
            )
            history_state = property_graph_history.state().to(device)
            transition["x_t"] = torch.cat(
                [transition["x_t"], history_state], dim=-1
            )
        x_t = transition["x_t"]
        edge_index_t = transition["edge_index_t"]
        topology_count_context = (
            _topology_count_context(
                edge_index_t,
                num_nodes=num_nodes,
                dtype=x_t.dtype,
            )
            if args.topology_count_context
            else None
        )
        current_edge_count = max(int(edge_index_t.shape[1]), 1)
        edge_weight_t = _edge_weight(transition, args.edge_input)
        if edge_weight_t is not None:
            edge_weight_t = edge_weight_t.to(device)
        operations, joint_action_groups_active = _action_groups_for_step(
            args.task,
            joint_action_groups=args.joint_action_groups,
            global_step=global_step,
            interval=args.joint_action_group_interval,
        )
        synthetic_intervention_active = bool(
            args.lambda_synthetic_intervention > 0.0









            and world_action_enabled
            and (
                not args.joint_action_groups
                or joint_action_groups_active
            )
        )
        controller_warmup_epochs = (
            args.world_warmup_epochs
            if args.controller_warmup_epochs is None
            else args.controller_warmup_epochs
        )
        grpo_step_active = bool(
            epoch >= controller_warmup_epochs
            and global_step % args.grpo_update_interval == 0
        )





        rollout_sampling_active = bool(
            grpo_step_active or synthetic_intervention_active
        )
        allocation = group_sampler.allocate(
            operations,
            total_budget=args.rollout_budget,
            min_per_group=args.min_group_size,
        )
        candidate_edges: dict[GraphOperation, torch.Tensor] = {}
        candidate_nodes: dict[GraphOperation, torch.Tensor] = {}
        transition_cache_key = int(transition_cpu["transition_id"])
        if args.task == "topology" or joint_action_groups_active:
            cached_device_candidates = (
                device_topology_candidate_cache.get(transition_cache_key)
                if device_topology_candidate_cache is not None
                else None
            )
            if cached_device_candidates is not None:
                candidate_edges = cached_device_candidates
                topology_candidates_cpu = None
            else:
                topology_candidates_cpu = None
            cached_topology_candidates = (
                deterministic_topology_candidate_cache.get(transition_cache_key)
                if deterministic_topology_candidate_cache is not None
                else None
            )
            deterministic_candidates = cached_topology_candidates is not None
            if cached_device_candidates is None and cached_topology_candidates is None:
                topology_candidates_cpu = _topology_candidates(
                    edge_index_t,
                    num_nodes=num_nodes,
                    metadata=metadata,
                    max_addition_candidates=args.max_addition_candidates,
                    generator=generator,
                    sample_without_materializing=joint_action_groups_active,
                    candidate_policy=args.topology_candidate_policy,
                    seen_codes=(
                        topology_pair_state.observed_codes()
                        if args.topology_candidate_policy == "seen_pairs"
                        and topology_pair_state is not None
                        else None
                    ),
                )




                addition_count = int(
                    topology_candidates_cpu[GraphOperation.ADD_EDGE].shape[1]
                )
                deterministic_candidates = (
                    args.max_addition_candidates <= 0
                    or addition_count < args.max_addition_candidates
                )
                if (
                    deterministic_topology_candidate_cache is not None
                    and deterministic_candidates
                ):
                    cached_topology_candidates = {
                        operation: edges.detach().cpu()
                        for operation, edges in topology_candidates_cpu.items()
                    }
                    deterministic_topology_candidate_cache[
                        transition_cache_key
                    ] = cached_topology_candidates
            elif cached_device_candidates is None:
                topology_candidates_cpu = cached_topology_candidates
            if cached_device_candidates is None:
                assert topology_candidates_cpu is not None
                candidate_edges = {
                    operation: edges.to(device)
                    for operation, edges in topology_candidates_cpu.items()
                }
                if (
                    device_topology_candidate_cache is not None
                    and deterministic_candidates
                ):
                    device_topology_candidate_cache[transition_cache_key] = (
                        candidate_edges
                    )
            if (
                args.task == "topology"
                and args.topology_candidate_policy == "seen_pairs"
            ):





                legal_add = candidate_edges[GraphOperation.ADD_EDGE]
                legal_codes = (
                    legal_add[0] * num_nodes + legal_add[1]
                    if legal_add.numel()
                    else torch.empty(0, dtype=torch.long, device=device)
                )
                target_add = transition["edge_added"].to(device)
                target_codes = (
                    target_add[0] * num_nodes + target_add[1]
                    if target_add.numel()
                    else torch.empty(0, dtype=torch.long, device=device)
                )
                transition = dict(transition)
                transition["edge_added"] = target_add[
                    :, torch.isin(target_codes, legal_codes)
                ]
        if args.task == "node_property":






            observed_property_nodes = torch.unique(
                transition_cpu["property_node_ids_t"].to(
                    device=device, dtype=torch.long
                )
            )
            candidate_nodes[GraphOperation.MODIFY_NODE_PROPERTY] = (
                observed_property_nodes
            )
        cached_pair_context = (
            device_topology_pair_context_cache.get(transition_cache_key)
            if device_topology_pair_context_cache is not None
            and transition_cache_key in (
                device_topology_candidate_cache or {}
            )
            else None
        )
        topology_pair_context = (
            cached_pair_context
            if cached_pair_context is not None
            else (
                _topology_pair_features(
                    topology_pair_state,
                    candidate_edges,
                    time_index=transition_index,
                )
                if args.task == "topology"
                else {}
            )
        )
        if (
            cached_pair_context is None
            and device_topology_pair_context_cache is not None
            and transition_cache_key in (device_topology_candidate_cache or {})
        ):
            device_topology_pair_context_cache[transition_cache_key] = (
                topology_pair_context
            )

        profile_setup = profile_mark()

        model.eval()
        action_encoder.eval()
        controller.eval()






        with _preserve_world_rng(device), torch.no_grad():
            z_policy_raw = model.encode_observed_graph(
                x_t,
                edge_index_t,
                edge_weight_t,
                topology_cache_key=int(transition_cpu["transition_id"]),
            )
            z_policy = model._normalize_latent(z_policy_raw)
        node_weights = structure_node_weights(
            edge_index_t,
            num_nodes=num_nodes,
            historical_change_counts=historical_change_counts,
            history_steps=transition_index,
        )
        topology_context = (
            _topology_reward_context(
                transition,
                candidate_edges,
                node_weights,
                device=device,
            )
            if args.task == "topology"
            else None
        )
        profile_policy = profile_mark()
        rollouts: list[RewardedSequenceRollout] = []
        reward_details: list[dict[str, float]] = []
        rollout_target_z: torch.Tensor | None = None




        with _preserve_world_rng(device), torch.no_grad():
            controller_node_state = controller.node_state(z_policy.detach(), hidden)
            valid_rollout_operations = tuple(
                operation
                for operation in operations
                if operation
                not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
                or (
                    operation in candidate_edges
                    and candidate_edges[operation].shape[1] > 0
                )
            )
            if not rollout_sampling_active:
                samples = []
            elif (
                args.task == "topology"
                and args.batch_topology_grpo_rollouts
                and valid_rollout_operations
                and all(
                    operation
                    in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
                    for operation in valid_rollout_operations
                )
            ):
                rollout_groups = [
                    operation
                    for operation in valid_rollout_operations
                    for _ in range(allocation[operation])
                ]
                samples = controller.sample_mixed_edge_action_sequences_batch(
                    z_policy.detach(),
                    hidden,
                    rollout_groups,
                    candidate_edges=candidate_edges,
                    valid_operations=valid_rollout_operations,
                    min_actions=args.min_actions,
                    max_actions=args.max_actions,
                    node_state=controller_node_state,
                )
            elif (
                len(operations) == 1
                and operations[0]
                not in {GraphOperation.ADD_EDGE, GraphOperation.REMOVE_EDGE}
            ):
                samples = controller.sample_node_action_sequences_batch(
                    z_policy.detach(),
                    hidden,
                    operations[0],
                    candidate_nodes=candidate_nodes.get(operations[0]),
                    min_actions=args.min_actions,
                    max_actions=args.max_actions,
                    sample_magnitude=(
                        args.task != "node_change" and args.sample_action_magnitude
                    ),
                    node_state=controller_node_state,
                    batch_size=allocation[operations[0]],
                )
            else:
                samples = [
                    controller.sample_action_sequence(
                        z_policy.detach(),
                        hidden,
                        operation,
                        candidate_edges=candidate_edges,
                        candidate_nodes=candidate_nodes,
                    valid_operations=valid_rollout_operations,
                    min_actions=args.min_actions,
                    max_actions=args.max_actions,
                    sample_magnitude=(
                        args.task != "node_change" and args.sample_action_magnitude
                    ),
                        node_state=controller_node_state,
                    )
                    for operation in valid_rollout_operations
                    for _ in range(allocation[operation])
                ]
            profile_rollout_sample = profile_mark()
            if samples:
                encoded_actions = torch.stack(
                    [
                        _encode_action_or_zero(
                            action_encoder,
                            z_policy,
                            sample.action,
                            enabled=world_action_enabled,
                        )
                        for sample in samples
                    ]
                )
                batched_outputs = _batched_action_model_outputs(
                    model,
                    x_t,
                    edge_index_t,
                    hidden,
                    encoded_actions,
                    edge_weight_t=edge_weight_t,
                    addition_candidate_edges=candidate_edges.get(GraphOperation.ADD_EDGE),
                    deletion_candidate_edges=candidate_edges.get(GraphOperation.REMOVE_EDGE),
                    addition_pair_features=topology_pair_context.get(GraphOperation.ADD_EDGE),
                    deletion_pair_features=topology_pair_context.get(GraphOperation.REMOVE_EDGE),
                    decode_node_state=(
                        args.task != "topology" or joint_action_groups_active
                    ),
                    precomputed_z_t_raw=z_policy_raw,
                )
                if args.task == "topology":



                    reward_edge_weight = (
                        transition.get("edge_weight_next")
                        if args.edge_input == "weighted"
                        else None
                    )
                    rollout_target_z = model.encode_target(
                        transition["x_next"],
                        transition["edge_index_next"],
                        edge_weight_next=reward_edge_weight,
                        topology_cache_key=int(transition_cpu["transition_id"]) + 1,
                    )
                batch_topology_rewards = (
                    args.task == "topology"
                    and args.batch_topology_grpo_rewards
                    and topology_context is not None
                    and rollout_target_z is not None
                    and args.topology_action_reward_weight == 0.0
                )
                if batch_topology_rewards:




                    reward_values = _batched_topology_decoded_rewards(
                        samples,
                        batched_outputs,
                        transition,
                        node_weights,
                        candidate_edges=candidate_edges,
                        topology_context=topology_context,
                        topology_exact_reward_weight=(
                            args.topology_exact_reward_weight
                        ),
                        topology_target_z=rollout_target_z,
                        topology_composite_embedding_reward=(
                            args.topology_composite_embedding_reward
                        ),
                    ).detach().cpu().tolist()
                    for sample, reward in zip(samples, reward_values):
                        scalar_reward = float(reward)
                        rollouts.append(
                            RewardedSequenceRollout(sample, scalar_reward)
                        )
                        reward_details.append(
                            {"combined_reward": scalar_reward}
                        )
                else:
                    for rollout_index, sample in enumerate(samples):
                        outputs = {
                            key: value[rollout_index]
                            for key, value in batched_outputs.items()
                        }
                        reward, details = _decoded_reward(
                            args.task,
                            sample.action,
                            outputs,
                            transition,
                            node_weights,
                            candidate_edges=candidate_edges,
                            candidate_nodes=candidate_nodes,
                            device=device,
                            topology_exact_reward_weight=args.topology_exact_reward_weight,
                            topology_context=topology_context,
                            topology_target_z=rollout_target_z,
                            topology_composite_embedding_reward=(
                                args.topology_composite_embedding_reward
                            ),
                            topology_action_reward_weight=(
                                args.topology_action_reward_weight
                            ),
                            node_action_reward_weight=args.node_action_reward_weight,
                            node_action_value_reward_weight=(
                                args.node_action_value_reward_weight
                            ),
                            property_mode=property_mode,
                            property_change_threshold=property_change_threshold,
                        )
                        rollouts.append(RewardedSequenceRollout(sample, reward))
                        reward_details.append(details)

        profile_rollout = profile_mark()



        world_rollouts = _select_diverse_world_rollouts(
            rollouts, args.world_intervention_rollouts
        )
        model.train()
        action_encoder.train()





        z_world_raw = model.encode_observed_graph(
            x_t,
            edge_index_t,
            edge_weight_t,
            topology_cache_key=int(transition_cpu["transition_id"]),
        )
        profile_world_encode = profile_mark()


        with torch.no_grad():
            forecast_sequence = controller.greedy_action_sequence(
                z_policy.detach(),
                hidden,
                candidate_edges=candidate_edges,
                candidate_nodes=candidate_nodes,
                valid_operations=TASK_OPERATIONS[args.task],
                min_actions=args.min_actions,
                max_actions=args.max_actions,
                predict_magnitude=(
                    args.task not in {"topology", "node_change"}
                    and args.sample_action_magnitude
                ),
                stop_threshold=args.inference_stop_threshold,
            )
        z_world = model._normalize_latent(z_world_raw)



        forecast_controller_node_state = controller_node_state
        forecast_action_confidence = z_world.new_zeros(())
        action_plan_target: torch.Tensor | None = None
        action_value_target: torch.Tensor | None = None
        action_value_probability: torch.Tensor | None = None
        if (
            args.task == "topology"
            and world_action_enabled
            and args.soft_edge_action_conditioning
        ):



            forecast_log_counts = (
                controller.topology_log_counts(
                    forecast_controller_node_state,
                    context=topology_count_context,
                )
                if args.controller_topology_count_weight > 0.0
                else None
            )
            if (
                forecast_log_counts is not None
                and args.topology_count_target_mode == "edge_rate"
            ):



                forecast_counts = torch.expm1(
                    forecast_log_counts.clamp(0.0, 12.0)
                ) * float(current_edge_count)
                forecast_log_counts = torch.log1p(forecast_counts).clamp(0.0, 12.0)
            forecast_support_count_by_operation = None
            if (
                args.soft_edge_action_count_support
                and forecast_log_counts is not None
            ):





                forecast_count = torch.expm1(
                    forecast_log_counts.detach().clamp(0.0, 12.0)
                ).round().to(torch.long)
                forecast_support_count_by_operation = {
                    GraphOperation.ADD_EDGE: min(
                        max(int(forecast_count[0].item()), 0),
                        int(candidate_edges[GraphOperation.ADD_EDGE].shape[1]),
                    ),
                    GraphOperation.REMOVE_EDGE: min(
                        max(int(forecast_count[1].item()), 0),
                        int(candidate_edges[GraphOperation.REMOVE_EDGE].shape[1]),
                    ),
                }
            forecast_edge_probability = controller.soft_edge_action_probabilities(
                forecast_controller_node_state,
                candidate_edges,
                chunk_size=65536,
                top_k_per_operation=args.soft_edge_action_topk,
                support_count_by_operation=forecast_support_count_by_operation,
                mode=args.soft_edge_action_mode,
            )
            encoded_forecast_action = action_encoder.encode_soft_edge_actions(
                z_world,
                candidate_edges,
                forecast_edge_probability,
                chunk_size=65536,
                confidence_gate=args.confidence_gated_soft_action,
                operation_log_counts=(
                    {
                        GraphOperation.ADD_EDGE: forecast_log_counts[0],
                        GraphOperation.REMOVE_EDGE: forecast_log_counts[1],
                    }
                    if forecast_log_counts is not None
                    else None
                ),
            )
            if (
                args.action_plan_consistency_weight > 0.0
                or args.action_decoder_consistency_weight > 0.0
            ):







                action_plan_target = encoded_forecast_action[
                    "target_probability_by_operation"
                ]
            forecast_action = encoded_forecast_action["nodewise"]
            forecast_action_confidence = encoded_forecast_action["confidence"]
        elif (
            args.task in {"node_change", "node_state", "node_property"}
            and world_action_enabled
            and args.soft_node_action_conditioning
        ):
            soft_operation = (
                GraphOperation.MODIFY_NODE_PROPERTY
                if args.task == "node_property"
                else GraphOperation.MODIFY_NODE_STATE
            )
            forecast_probability = calibrated_soft_node_action_probability(
                controller.node_target_head(
                    forecast_controller_node_state
                ).squeeze(-1),
                historical_positive_count=historical_change_counts.sum(),
                historical_observation_count=historical_change_observations,
                mode=args.soft_node_action_probability_calibration,
                smoothing=args.soft_node_action_prior_smoothing,
            )
            if args.task == "node_property":




                observed_nodes = candidate_nodes.get(soft_operation)
                if observed_nodes is not None:
                    observed_mask = torch.zeros_like(forecast_probability)
                    observed_mask[observed_nodes] = 1.0
                    forecast_probability = forecast_probability * observed_mask
            forecast_node_support_count = None
            if (
                args.soft_node_action_count_support
                and args.controller_node_count_weight > 0.0
            ):





                predicted_count = torch.expm1(
                    controller.node_log_count(
                        forecast_controller_node_state
                    ).detach().clamp(0.0, 12.0)
                ).round().to(torch.long)
                eligible_nodes = candidate_nodes.get(soft_operation)
                maximum_count = (
                    int(eligible_nodes.numel())
                    if eligible_nodes is not None
                    else int(forecast_probability.numel())
                )
                forecast_node_support_count = min(
                    max(int(predicted_count.item()), 0), maximum_count
                )
            forecast_probability = forecast_probability.detach()
            forecast_value = None
            if (
                args.task in {"node_state", "node_property"}
                and args.soft_node_magnitude_conditioning
            ):
                forecast_value = controller._magnitude_distribution(
                    forecast_controller_node_state, soft_operation
                )[0]
                forecast_value = forecast_value.detach()
            if (
                args.task == "node_state"
                and args.action_value_decoder_consistency_weight > 0.0
            ):
                if forecast_value is None:
                    raise RuntimeError(
                        "Action-value decoder consistency requires "
                        "--soft_node_magnitude_conditioning."
                    )





                action_value_target = forecast_value
                action_value_probability = forecast_probability
            encoded_forecast_action = action_encoder.encode_soft_node_action(
                z_world,
                forecast_probability,
                soft_operation,
                node_value=forecast_value,
                change_gated_value=args.change_gated_action_value,
                confidence_gate=args.confidence_gated_soft_action,
                localization_mode=args.soft_node_action_mode,
                top_k=args.soft_node_action_topk,
                support_count=forecast_node_support_count,
            )
            if (
                args.action_plan_consistency_weight > 0.0
                or args.action_decoder_consistency_weight > 0.0
                and world_action_enabled
            ):



                action_plan_target = encoded_forecast_action[
                    "target_probability"
                ]
            forecast_action = encoded_forecast_action["nodewise"]
            forecast_action_confidence = encoded_forecast_action["confidence"]
        else:
            forecast_action = _encode_action_or_zero(
                action_encoder,
                z_world,
                forecast_sequence.action,
                enabled=world_action_enabled,
            )
        profile_action = profile_mark()







        forecast_candidate_edges = candidate_edges
        forecast_topology_context = topology_context
        forecast_pair_context = topology_pair_context
        if args.task == "topology":
            forecast_candidate_edges, forecast_topology_context = (
                _topology_decoder_supervision_candidates(
                    transition,
                    candidate_edges,
                    num_nodes=num_nodes,
                    node_weights=node_weights,
                    addition_neg_ratio=args.topology_addition_neg_ratio,
                    addition_negative_strategy=args.topology_addition_negative_strategy,
                    generator=generator,
                )
            )
            forecast_pair_context = _topology_pair_features(
                topology_pair_state,
                forecast_candidate_edges,
                time_index=transition_index,
            )
        profile_supervision = profile_mark()




        advantage_rng_cpu: torch.Tensor | None = None
        advantage_rng_cuda: torch.Tensor | None = None
        if (
            (
                args.action_transition_advantage_weight > 0.0
                or args.action_observable_advantage_weight > 0.0
                or args.zero_action_anchor_weight > 0.0
                or args.action_residual_only_world_loss
            )
            and world_action_enabled
        ):
            advantage_rng_cpu = torch.get_rng_state().clone()
            if device.type == "cuda":
                advantage_rng_cuda = torch.cuda.get_rng_state(device).clone()
        forecast_outputs = model(
            x_t,
            edge_index_t,
            hidden,
            action=forecast_action,
            edge_weight_t=edge_weight_t,
            addition_candidate_edges=forecast_candidate_edges.get(
                GraphOperation.ADD_EDGE
            ),
            deletion_candidate_edges=forecast_candidate_edges.get(
                GraphOperation.REMOVE_EDGE
            ),
            addition_pair_features=forecast_pair_context.get(GraphOperation.ADD_EDGE),
            deletion_pair_features=forecast_pair_context.get(GraphOperation.REMOVE_EDGE),
            decode_node_state=args.task != "topology",
            commit_history=False,
            precomputed_z_t_raw=z_world_raw,
        )
        profile_forecast = profile_mark()
        observed_target_z: torch.Tensor | None = rollout_target_z
        target_graph_cpu: dict[str, Any] | None = None
        if "x_next" in transition and "edge_index_next" in transition:
            target_graph_cpu = {
                "x_next": transition["x_next"],
                "edge_index_next": transition["edge_index_next"],
                "edge_weight_next": transition.get("edge_weight_next"),
            }
        elif hasattr(transitions, "target_graph"):



            target_graph_cpu = transitions.target_graph(transition_index)







        latent_train_allowed = bool(transition_cpu.get("latent_train_allowed", True))
        if (
            observed_target_z is None
            and
            latent_train_allowed
            and target_graph_cpu
            and "x_next" in target_graph_cpu
        ):
            observed_x_next = target_graph_cpu["x_next"].to(device)
            if property_graph_history is not None:
                observed_history_next = property_graph_history.hypothetical_state_after(
                    transition_cpu["property_node_ids"],
                    transition_cpu["property_target"],
                ).to(device)
                observed_x_next = torch.cat(
                    [observed_x_next, observed_history_next], dim=-1
                )
            observed_edge_weight = (
                target_graph_cpu.get("edge_weight_next")
                if args.edge_input == "weighted"
                else None
            )
            if observed_edge_weight is not None:
                observed_edge_weight = observed_edge_weight.to(device)
            observed_target_z = model.encode_target(
                observed_x_next,
                target_graph_cpu["edge_index_next"].to(device),
                edge_weight_next=observed_edge_weight,
                topology_cache_key=int(transition_cpu["transition_id"]) + 1,
            )
        profile_target = profile_mark()
        forecast_terms = _observed_transition_loss(
            args.task,
            forecast_outputs,
            transition,
            target_z=observed_target_z,
            topology_context=forecast_topology_context,
            property_mode=property_mode,
            latent_weight=args.forecast_latent_weight,
            change_weight=args.forecast_change_weight,
            change_pos_weight=args.forecast_change_pos_weight,
            state_weight=args.forecast_state_weight,
            state_loss_mode=args.forecast_state_loss_mode,
            state_loss_type=args.forecast_state_loss_type,
            property_loss_type=args.property_loss_type,
            property_change_threshold=property_change_threshold,
            latent_cosine_weight=args.forecast_latent_cosine_weight,
            latent_variance_weight=args.forecast_latent_variance_weight,
            all_state_weight=args.forecast_all_state_weight,
            current_state_weight=args.forecast_current_state_weight,
            topology_count_weight=args.topology_count_weight,
            topology_decoder_loss=args.topology_decoder_loss,
            device=device,
            action_plan_target=action_plan_target,
            action_plan_consistency_weight=(
                args.action_plan_consistency_weight
                if action_plan_target is not None
                else 0.0
            ),
            action_plan_positive_weight_cap=args.action_plan_positive_weight_cap,
            action_decoder_consistency_weight=(
                args.action_decoder_consistency_weight
                if action_plan_target is not None
                else 0.0
            ),
            action_decoder_positive_weight_cap=(
                args.action_decoder_positive_weight_cap
                if action_plan_target is not None
                else 0.0
            ),
            action_value_decoder_consistency_weight=(
                args.action_value_decoder_consistency_weight
                if action_value_target is not None
                else 0.0
            ),
            action_value_target=action_value_target,
            action_value_probability=action_value_probability,
        )











        controller_state_alignment_loss = forecast_terms["total"].new_zeros(())
        if (
            args.controller_state_alignment_weight > 0.0
            and args.task in {"node_change", "node_state"}
        ):
            alignment_state = controller.node_state(z_world, hidden.detach())
            alignment_logits = controller.node_target_head(alignment_state).squeeze(-1)
            alignment_labels = transition["node_changed"].to(
                device=device, dtype=alignment_logits.dtype
            )
            alignment_positives = alignment_labels.sum()
            alignment_negatives = alignment_labels.numel() - alignment_positives
            alignment_pos_weight = (
                alignment_negatives / alignment_positives.clamp_min(1.0)
            ).clamp(1.0, 20.0)
            if alignment_labels.numel():
                controller_state_alignment_loss = F.binary_cross_entropy_with_logits(
                    alignment_logits,
                    alignment_labels,
                    pos_weight=alignment_pos_weight,
                )
            if (
                args.task == "node_state"
                and args.controller_state_alignment_magnitude_weight > 0.0
                and bool(alignment_labels.to(torch.bool).any())
            ):
                alignment_mu, alignment_logstd = controller._magnitude_distribution(
                    alignment_state,
                    GraphOperation.MODIFY_NODE_STATE,
                )
                alignment_changed = alignment_labels.to(torch.bool)
                alignment_target_delta = transition["node_delta"].to(
                    device=device, dtype=alignment_mu.dtype
                )
                if args.controller_magnitude_loss == "gaussian_nll":
                    alignment_error = (
                        alignment_target_delta[alignment_changed]
                        - alignment_mu[alignment_changed]
                    ) * torch.exp(-alignment_logstd[alignment_changed])
                    alignment_magnitude_loss = (
                        alignment_logstd[alignment_changed]
                        + 0.5 * alignment_error.square()
                    ).mean()
                else:
                    alignment_magnitude_loss = F.smooth_l1_loss(
                        alignment_mu[alignment_changed],
                        alignment_target_delta[alignment_changed],
                    )
                controller_state_alignment_loss = (
                    controller_state_alignment_loss
                    + float(args.controller_state_alignment_magnitude_weight)
                    * alignment_magnitude_loss
                )
        action_transition_advantage = forecast_terms["total"].new_zeros(())
        action_transition_mse = forecast_terms["total"].new_zeros(())
        zero_action_transition_mse = forecast_terms["total"].new_zeros(())
        action_transition_advantage_active = False
        action_observable_advantage = forecast_terms["total"].new_zeros(())
        action_observable_value = forecast_terms["total"].new_zeros(())
        zero_action_observable_value = forecast_terms["total"].new_zeros(())
        action_observable_advantage_active = False
        zero_action_outputs: dict[str, torch.Tensor] | None = None
        zero_action_terms: dict[str, torch.Tensor] | None = None
        zero_action_loss_required = bool(
            args.action_transition_advantage_weight > 0.0
            or args.action_observable_advantage_weight > 0.0
            or args.zero_action_anchor_weight > 0.0
            or args.action_residual_only_world_loss
        )
        zero_action_hidden_required = False
        if (
            (zero_action_loss_required or zero_action_hidden_required)
            and world_action_enabled
            and advantage_rng_cpu is not None
        ):


            rng_devices: list[int] = []
            if device.type == "cuda":
                rng_devices = [
                    torch.cuda.current_device()
                    if device.index is None
                    else int(device.index)
                ]
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.set_rng_state(advantage_rng_cpu)
                if advantage_rng_cuda is not None:
                    torch.cuda.set_rng_state(advantage_rng_cuda, device=device)
                with torch.set_grad_enabled(
                    args.zero_action_anchor_weight > 0.0
                    or args.action_residual_only_world_loss
                ):
                    zero_action_outputs = model(
                        x_t,
                        edge_index_t,
                        hidden,
                        action=forecast_action.new_zeros(forecast_action.shape),
                        edge_weight_t=edge_weight_t,
                        addition_candidate_edges=forecast_candidate_edges.get(
                            GraphOperation.ADD_EDGE
                        ),
                        deletion_candidate_edges=forecast_candidate_edges.get(
                            GraphOperation.REMOVE_EDGE
                        ),
                        addition_pair_features=forecast_pair_context.get(
                            GraphOperation.ADD_EDGE
                        ),
                        deletion_pair_features=forecast_pair_context.get(
                            GraphOperation.REMOVE_EDGE
                        ),
                        decode_node_state=args.task != "topology",
                        commit_history=False,
                        precomputed_z_t_raw=z_world_raw,
                    )
            if zero_action_loss_required:





                zero_action_terms = _observed_transition_loss(
                    args.task,
                    zero_action_outputs,
                    transition,
                    target_z=observed_target_z,
                    topology_context=forecast_topology_context,
                    property_mode=property_mode,
                    latent_weight=args.forecast_latent_weight,
                    change_weight=args.forecast_change_weight,
                    change_pos_weight=args.forecast_change_pos_weight,
                    state_weight=args.forecast_state_weight,
                    state_loss_mode=args.forecast_state_loss_mode,
                    state_loss_type=args.forecast_state_loss_type,
                    property_loss_type=args.property_loss_type,
                    property_change_threshold=property_change_threshold,
                    latent_cosine_weight=args.forecast_latent_cosine_weight,
                    latent_variance_weight=args.forecast_latent_variance_weight,
                    all_state_weight=args.forecast_all_state_weight,
                    current_state_weight=args.forecast_current_state_weight,
                    topology_count_weight=args.topology_count_weight,
                    topology_decoder_loss=args.topology_decoder_loss,
                    device=device,
                )
            if (
                args.action_transition_advantage_weight > 0.0
                and observed_target_z is not None
            ):
                advantage_terms = _action_transition_advantage_loss(
                    forecast_outputs["latent_mu"],
                    zero_action_outputs["latent_mu"],
                    observed_target_z,
                    margin=args.action_transition_advantage_margin,
                )
                action_transition_advantage = advantage_terms["total"]
                action_transition_mse = advantage_terms["action_mse"]
                zero_action_transition_mse = advantage_terms["zero_action_mse"]
                action_transition_advantage_active = True
            if args.action_observable_advantage_weight > 0.0:
                observable_advantage_terms = _action_observable_advantage_loss(
                    args.task,
                    forecast_terms,
                    zero_action_terms,
                    margin=args.action_observable_advantage_margin,
                )
                action_observable_advantage = observable_advantage_terms["total"]
                action_observable_value = observable_advantage_terms[
                    "action_observable"
                ]
                zero_action_observable_value = observable_advantage_terms[
                    "zero_action_observable"
                ]
                action_observable_advantage_active = True
        action_transition_advantage_grads: tuple[torch.Tensor | None, ...] | None = None
        if action_transition_advantage_active:



            action_transition_advantage_grads = torch.autograd.grad(
                float(args.action_transition_advantage_weight)
                * action_transition_advantage,
                action_advantage_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        action_observable_advantage_grads: tuple[torch.Tensor | None, ...] | None = None
        if action_observable_advantage_active:




            action_observable_advantage_grads = torch.autograd.grad(
                float(args.action_observable_advantage_weight)
                * action_observable_advantage,
                action_advantage_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        intervention_losses: list[torch.Tensor] = []
        latent_losses: list[torch.Tensor] = []
        observable_losses: list[torch.Tensor] = []
        action_locality_losses: list[torch.Tensor] = []
        if synthetic_intervention_active:




            effective_intervention_actions = [
                bounded_graph_edit_sequence(
                    rollout.sample.action,
                    max_abs_delta=args.synthetic_action_delta_scale,
                )
                for rollout in world_rollouts
            ]
            intervention_actions = torch.stack(
                [
                    _encode_action_or_zero(
                        action_encoder,
                        z_policy,
                        effective_intervention_actions[index],
                        enabled=world_action_enabled,
                    )
                    for index, rollout in enumerate(world_rollouts)
                ]
            )
            batched_intervention_outputs = _batched_action_model_outputs(
                model,
                x_t,
                edge_index_t,
                hidden,
                intervention_actions,
                edge_weight_t=edge_weight_t,
                addition_candidate_edges=candidate_edges.get(GraphOperation.ADD_EDGE),
                deletion_candidate_edges=candidate_edges.get(GraphOperation.REMOVE_EDGE),
                addition_pair_features=topology_pair_context.get(GraphOperation.ADD_EDGE),
                deletion_pair_features=topology_pair_context.get(GraphOperation.REMOVE_EDGE),
                decode_node_state=(
                    args.task != "topology" or joint_action_groups_active
                ),
                precomputed_z_t_raw=z_world_raw,
            )
            interventions = [
                apply_graph_edit_sequence(
                    x_t=x_t,
                    edge_index_t=edge_index_t,
                    edge_weight_t=edge_weight_t,
                    action=effective_intervention_actions[index],
                    property_slice=transition.get("property_slice"),
                    property_update_mode=(
                        "logit_shift"
                        if args.task == "node_property"
                        and property_mode.startswith("logit_")
                        else "additive"
                    ),
                    undirected_edges=False,
                )
                for index, rollout in enumerate(world_rollouts)
            ]



            batched_target_z: torch.Tensor | None = None
            same_topology = bool(interventions) and all(
                torch.equal(intervention["edge_index"], edge_index_t)
                for intervention in interventions
            )
            if args.task != "topology" and same_topology:
                intervention_x = torch.stack(
                    [intervention["x"] for intervention in interventions]
                )
                batched_target_z = vmap(
                    lambda x_next: model.encode_target(
                        x_next,
                        edge_index_t,
                        edge_weight_next=edge_weight_t,
                        topology_cache_key=int(transition_cpu["transition_id"]) + 1,
                    )
                )(intervention_x)
            for rollout_index, rollout in enumerate(world_rollouts):
                sample = rollout.sample
                effective_action = effective_intervention_actions[rollout_index]
                intervention = interventions[rollout_index]
                outputs = {
                    key: value[rollout_index]
                    for key, value in batched_intervention_outputs.items()
                }
                assert intervention["x"] is not None
                assert intervention["edge_index"] is not None
                if batched_target_z is not None:
                    target_z = batched_target_z[rollout_index]
                else:
                    target_z = model.encode_target(
                        intervention["x"],
                        intervention["edge_index"],
                        edge_weight_next=intervention["edge_weight"],
                    )
                latent_loss = latent_transition_terms(
                    outputs,
                    target_z,
                    cosine_weight=args.forecast_latent_cosine_weight,
                    variance_weight=args.forecast_latent_variance_weight,
                )["total"]
                observable_loss = _observable_intervention_loss(
                    args.task,
                    effective_action,
                    outputs,
                    intervention,
                    transition,
                    candidate_edges=candidate_edges,
                    device=device,
                    property_mode=property_mode,
                )
                action_locality_loss = _latent_action_locality_loss(
                    outputs,
                    effective_action,
                    num_nodes=num_nodes,
                    device=device,
                    margin=args.action_locality_margin,
                )
                latent_losses.append(latent_loss)
                observable_losses.append(observable_loss)
                action_locality_losses.append(action_locality_loss)
                intervention_losses.append(
                    latent_loss
                    + float(args.lambda_observable) * observable_loss
                    + float(args.lambda_action_locality) * action_locality_loss
                )
            synthetic_intervention_loss = torch.stack(intervention_losses).mean()
        else:
            synthetic_intervention_loss = forecast_terms["total"].new_zeros(())
        synthetic_intervention_grads: tuple[torch.Tensor | None, ...] | None = None
        if (
            synthetic_intervention_active
            and args.synthetic_intervention_action_only
            and action_response_parameters
        ):




            synthetic_intervention_grads = torch.autograd.grad(
                float(args.lambda_synthetic_intervention)
                * synthetic_intervention_loss,
                action_response_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        forecast_world_loss = forecast_terms["total"]
        zero_action_anchor_active = bool(
            args.zero_action_anchor_weight > 0.0
            and zero_action_terms is not None
        )
        if args.action_residual_only_world_loss and world_action_enabled:
            if zero_action_terms is None:
                raise RuntimeError(
                    "Residual-only World Model training requires a zero-action "
                    "counterfactual output."
                )





            forecast_world_loss = zero_action_terms["total"]
        elif zero_action_anchor_active:
            anchor_weight = float(args.zero_action_anchor_weight)
            forecast_world_loss = (
                forecast_world_loss
                + anchor_weight * zero_action_terms["total"]
            ) / (1.0 + anchor_weight)
        world_loss = (
            float(args.lambda_synthetic_intervention)
            * float(
                synthetic_intervention_active
                and not args.synthetic_intervention_action_only
            )
            * synthetic_intervention_loss
            + float(args.lambda_forecast) * forecast_world_loss
            + float(args.controller_state_alignment_weight)
            * controller_state_alignment_loss
        )
        profile_loss = profile_mark()
        world_optimizer.zero_grad(set_to_none=True)
        controller_optimizer.zero_grad(set_to_none=True)
        world_loss.backward()
        if action_transition_advantage_grads is not None:
            for parameter, gradient in zip(
                action_advantage_parameters, action_transition_advantage_grads
            ):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient.detach().clone()
                else:
                    parameter.grad.add_(gradient.detach())
        if action_observable_advantage_grads is not None:
            for parameter, gradient in zip(
                action_advantage_parameters, action_observable_advantage_grads
            ):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient.detach().clone()
                else:
                    parameter.grad.add_(gradient.detach())
        if synthetic_intervention_grads is not None:
            for parameter, gradient in zip(
                action_response_parameters, synthetic_intervention_grads
            ):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = gradient.detach().clone()
                else:
                    parameter.grad.add_(gradient.detach())



        for parameter in controller_parameters:
            parameter.grad = None
        world_grad_norm = torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), *action_encoder.parameters()], 5.0
        )
        world_optimizer.step()
        model.update_target_encoder()

        profile_world = profile_mark()

        controller.train()
        controller_grpo_node_state = controller.node_state(z_policy.detach(), hidden)
        if grpo_step_active:
            if not rollouts:
                raise RuntimeError("A scheduled GRPO update has no sampled rollouts.")
            controller_loss = grpo_sequence_loss(
                controller,
                rollouts,
                z_t=z_policy.detach(),
                h_t=hidden,
                candidate_edges=candidate_edges,
                candidate_nodes=candidate_nodes,
                valid_operations=valid_rollout_operations,
                max_actions=args.max_actions,
                clip_epsilon=args.grpo_clip,
                kl_coefficient=args.grpo_kl,
                entropy_coefficient=args.entropy_weight,
                node_state=controller_grpo_node_state,
                batch_mixed_edge_sequences=(
                    args.task == "topology"
                    and args.batch_topology_grpo_rollouts
                ),
            )
        else:
            zero = controller_grpo_node_state.new_zeros(())
            controller_loss = {
                "loss": zero,
                "objective": zero,
                "kl": zero,
            }





        controller_localization_loss = controller_loss["loss"].new_zeros(())
        controller_magnitude_loss = controller_loss["loss"].new_zeros(())
        controller_topology_count_loss = controller_loss["loss"].new_zeros(())
        controller_node_count_loss = controller_loss["loss"].new_zeros(())
        controller_supervised_loss = controller_loss["loss"].new_zeros(())
        if (
            args.task == "topology"
            and args.controller_supervised_weight > 0.0
        ):
            if topology_context is None:
                raise RuntimeError("Missing topology Controller supervision context.")
            (
                controller_localization_loss,
                controller_topology_count_loss,
            ) = _topology_controller_supervision_loss(
                controller,
                controller_grpo_node_state,
                candidate_edges,
                topology_context,
                transition,
                count_weight=args.controller_topology_count_weight,
                count_target_mode=args.topology_count_target_mode,
                count_context=topology_count_context,
            )
            controller_supervised_loss = (
                controller_localization_loss
                + float(args.controller_topology_count_weight)
                * controller_topology_count_loss
            )
        elif (
            args.task in {"node_change", "node_state", "node_property"}
            and args.controller_supervised_weight > 0.0
        ):
            controller_logits = controller.node_target_head(
                controller_grpo_node_state
            ).squeeze(-1)
            supervised_logits = controller_logits
            if args.task == "node_property":
                if property_change_threshold is None:
                    raise RuntimeError("Missing train-only property-change threshold.")
                property_ids = transition["property_node_ids"].to(
                    device=device, dtype=torch.long
                )
                comparable = transition["property_current_observed_mask"].to(
                    device=device, dtype=torch.bool
                )
                property_delta = (
                    transition["property_target"].to(device)
                    - transition["property_current_target"].to(device)
                )
                property_magnitude = property_delta.abs().mean(dim=-1)
                controller_labels = property_magnitude.ge(
                    float(property_change_threshold)
                ).to(controller_logits.dtype)
                supervised_logits = controller_logits.index_select(0, property_ids)
                supervised_logits = supervised_logits[comparable]
                controller_labels = controller_labels[comparable]
            else:
                controller_labels = transition["node_changed"].to(
                    device=device, dtype=controller_logits.dtype
                )






            if args.controller_node_count_weight > 0.0:
                target_log_count = torch.log1p(controller_labels.sum())
                controller_node_count_loss = F.smooth_l1_loss(
                    controller.node_log_count(controller_grpo_node_state),
                    target_log_count,
                )
            positives = controller_labels.sum()
            negatives = controller_labels.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
            if controller_labels.numel():
                controller_localization_loss = F.binary_cross_entropy_with_logits(
                    supervised_logits,
                    controller_labels,
                    pos_weight=pos_weight,
                )
            controller_supervised_loss = (
                controller_localization_loss
                + float(args.controller_node_count_weight)
                * controller_node_count_loss
            )
            if (
                args.task == "node_state"
                and args.controller_magnitude_supervised_weight > 0.0
                and bool(controller_labels.to(torch.bool).any())
            ):
                controller_mu, controller_logstd = controller._magnitude_distribution(
                    controller_grpo_node_state,
                    GraphOperation.MODIFY_NODE_STATE,
                )
                changed = controller_labels.to(torch.bool)
                target_delta = transition["node_delta"].to(
                    device=device, dtype=controller_mu.dtype
                )
                if args.controller_magnitude_loss == "gaussian_nll":
                    normalized_error = (
                        target_delta[changed] - controller_mu[changed]
                    ) * torch.exp(-controller_logstd[changed])
                    controller_magnitude_loss = (
                        controller_logstd[changed]
                        + 0.5 * normalized_error.square()
                    ).mean()
                else:




                    controller_magnitude_loss = F.smooth_l1_loss(
                        controller_mu[changed], target_delta[changed]
                    )
                controller_supervised_loss = (
                    controller_supervised_loss
                    + float(args.controller_magnitude_supervised_weight)
                    * controller_magnitude_loss
                )
            if (
                args.task == "node_property"
                and args.controller_magnitude_supervised_weight > 0.0
                and controller_labels.numel()
                and bool(controller_labels.to(torch.bool).any())
            ):
                controller_mu, controller_logstd = controller._magnitude_distribution(
                    controller_grpo_node_state,
                    GraphOperation.MODIFY_NODE_PROPERTY,
                )
                comparable_ids = property_ids[comparable]
                changed = controller_labels.to(torch.bool)
                predicted_mu = controller_mu.index_select(0, comparable_ids)[changed]
                predicted_logstd = controller_logstd.index_select(
                    0, comparable_ids
                )[changed]
                target_delta = property_delta[comparable][changed].to(
                    dtype=predicted_mu.dtype
                )
                if args.controller_magnitude_loss == "gaussian_nll":
                    normalized_error = (target_delta - predicted_mu) * torch.exp(
                        -predicted_logstd
                    )
                    controller_magnitude_loss = (
                        predicted_logstd + 0.5 * normalized_error.square()
                    ).mean()
                else:
                    controller_magnitude_loss = F.smooth_l1_loss(
                        predicted_mu, target_delta
                    )
                controller_supervised_loss = (
                    controller_supervised_loss
                    + float(args.controller_magnitude_supervised_weight)
                    * controller_magnitude_loss
                )
        grpo_active = grpo_step_active
        supervised_controller_active = (
            args.task in {
                "topology", "node_change", "node_state", "node_property"
            }
            and args.controller_supervised_weight > 0.0
        )
        controller_total_loss = (
            controller_loss["loss"]
            if grpo_active
            else controller_loss["loss"].new_zeros(())
        ) + float(args.controller_supervised_weight) * controller_supervised_loss






        controller_loss_requires_backward = bool(
            (grpo_active or supervised_controller_active)
            and controller_total_loss.requires_grad
        )
        controller_updated = controller_loss_requires_backward
        if controller_updated:
            if controller_loss_requires_backward:
                controller_total_loss.backward()
            controller_grad_norm = torch.nn.utils.clip_grad_norm_(
                controller_parameters, 5.0
            )
            controller_optimizer.step()
        else:
            controller_grad_norm = torch.zeros((), device=device)

        profile_grpo = profile_mark()









        commit_observation = getattr(model.state_model, "commit_observation", None)
        if not callable(commit_observation):
            raise RuntimeError(
                "The action-aware benchmark requires a state model with "
                "commit_observation()."
            )
        commit_observation(
            forecast_outputs["z_t"],
            action=forecast_action,
            graph_context=forecast_outputs["graph_context_t"],
        )
        if topology_pair_state is not None:


            topology_pair_state.observe(
                transition_cpu["edge_index_t"], time_index=transition_index
            )
        hidden = forecast_outputs["hidden_next"].detach()



        executed = forecast_sequence
        with torch.no_grad():
            _, executed_reward_details = _decoded_reward(
                args.task,
                executed.action,
                forecast_outputs,
                transition,
                node_weights,
                candidate_edges=forecast_candidate_edges,
                candidate_nodes=candidate_nodes,
                device=device,
                topology_exact_reward_weight=args.topology_exact_reward_weight,
                topology_context=forecast_topology_context,
                topology_target_z=observed_target_z,
                topology_composite_embedding_reward=(
                    args.topology_composite_embedding_reward
                ),
                topology_action_reward_weight=args.topology_action_reward_weight,
                node_action_reward_weight=args.node_action_reward_weight,
                node_action_value_reward_weight=(
                    args.node_action_value_reward_weight
                ),
                property_mode=property_mode,
                property_change_threshold=property_change_threshold,
            )
        profile_commit = profile_mark()





        group_history_operations = _observed_operation_groups(
            args.task, transition_cpu
        )
        executed_operations_by_transition[transition_index] = group_history_operations
        observed_operations_by_transition[transition_index] = group_history_operations
        observed_operations_prefix_index = max(
            observed_operations_prefix_index, transition_index + 1
        )
        group_sampler.observe(group_history_operations)
        historical_change_counts = historical_change_counts + _past_change_update(
            args.task, transition, device, num_nodes=num_nodes
        )
        historical_change_observations = (
            historical_change_observations
            + _past_change_observation_count(args.task, transition, device)
        )
        rewards = [rollout.reward for rollout in rollouts]
        diagnostic_keys = sorted(
            {key for details in reward_details for key in details}
        )




        mean_decoded_diagnostics = {}
        for key in diagnostic_keys:
            values = [
                float(details[key])
                for details in reward_details
                if isinstance(details.get(key), (int, float))
            ]
            if values:
                mean_decoded_diagnostics[key] = float(sum(values) / len(values))
        record = {
            "step": global_step,
            "epoch": epoch,
            "epoch_position": epoch_position,
            "transition_index": transition_index,
            "transition_id": int(transition["transition_id"]),
            "world_intervention_loss": float(world_loss.detach().item()),
            "synthetic_intervention_loss": float(
                synthetic_intervention_loss.detach().item()
            ),
            "observed_forecast_loss": float(
                forecast_terms["total"].detach().item()
            ),
            "observed_forecast_latent_loss": float(
                forecast_terms["latent"].detach().item()
            ),
            "observed_forecast_change_loss": float(
                forecast_terms["change"].detach().item()
            ),
            "observed_forecast_observable_loss": float(
                forecast_terms["observable"].detach().item()
            ),
            "observed_current_state_loss": float(
                forecast_terms["current_state"].detach().item()
            ),
            "controller_state_alignment_loss": float(
                controller_state_alignment_loss.detach().item()
            ),
            "action_plan_consistency_loss": float(
                forecast_terms["action_plan"].detach().item()
            ),
            "action_decoder_consistency_loss": float(
                forecast_terms["action_decoder_consistency"].detach().item()
            ),
            "action_value_decoder_consistency_loss": float(
                forecast_terms["action_value_decoder_consistency"].detach().item()
            ),
            "action_transition_advantage_loss": float(
                action_transition_advantage.detach().item()
            ),
            "action_transition_latent_mse": float(
                action_transition_mse.detach().item()
            ),
            "zero_action_transition_latent_mse": float(
                zero_action_transition_mse.detach().item()
            ),
            "action_observable_advantage_loss": float(
                action_observable_advantage.detach().item()
            ),
            "action_observable_value": float(
                action_observable_value.detach().item()
            ),
            "zero_action_observable_value": float(
                zero_action_observable_value.detach().item()
            ),
            "zero_action_anchor_loss": float(
                zero_action_terms["total"].detach().item()
                if zero_action_anchor_active and zero_action_terms is not None
                else 0.0
            ),
            "latent_intervention_loss": float(
                torch.stack(latent_losses).mean().detach().item()
                if latent_losses
                else 0.0
            ),
            "observable_intervention_loss": float(
                torch.stack(observable_losses).mean().detach().item()
                if observable_losses
                else 0.0
            ),
            "action_locality_loss": float(
                torch.stack(action_locality_losses).mean().detach().item()
                if action_locality_losses
                else 0.0
            ),
            "grpo_loss": float(controller_loss["loss"].detach().item()),
            "controller_supervised_loss": float(
                controller_supervised_loss.detach().item()
            ),
            "controller_localization_loss": float(
                controller_localization_loss.detach().item()
            ),
            "controller_magnitude_loss": float(
                controller_magnitude_loss.detach().item()
            ),
            "controller_topology_count_loss": float(
                controller_topology_count_loss.detach().item()
            ),
            "controller_node_count_loss": float(
                controller_node_count_loss.detach().item()
            ),
            "controller_total_loss": float(controller_total_loss.detach().item()),
            "grpo_objective": float(controller_loss["objective"].detach().item()),
            "grpo_kl": float(controller_loss["kl"].detach().item()),
            "world_grad_norm": float(world_grad_norm.detach().item()),
            "controller_grad_norm": float(controller_grad_norm.detach().item()),
            "controller_updated": controller_updated,
            "grpo_sampled": bool(rollouts),
            "grpo_step_active": grpo_step_active,
            "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "reward_std": float(
                torch.tensor(rewards, dtype=torch.float32).std(unbiased=False).item()
                if rewards
                else 0.0
            ),
            "nonzero_reward_rollouts": int(sum(reward > 0 for reward in rewards)),
            "mean_action_count": float(
                sum(len(rollout.sample.action) for rollout in rollouts) / len(rollouts)
            ) if rollouts else 0.0,
            "operation_allocation": {
                operation.name: count for operation, count in allocation.items()
            },
            "dynamic_group_observed_operations": [
                operation.name for operation in group_history_operations
            ],
            "joint_action_groups_active": joint_action_groups_active,


            "synthetic_topology_active": joint_action_groups_active,
            "executed_selection_rule": "future_free_greedy_policy",
            "executed_policy_log_probability": float(
                executed.old_log_prob.detach().item()
            ),
            "executed_action_count": len(executed.action),
            "forecast_action_confidence": float(
                forecast_action_confidence.detach().item()
            ),
            "forecast_action_mean_norm": float(
                forecast_action.detach().norm(dim=-1).mean().item()
            ),
            "action_residual_scale": float(
                torch.as_tensor(
                    model.state_model.action_residual_scale(),
                    device=device,
                )
                .detach()
                .item()
                if hasattr(model.state_model, "action_residual_scale")
                else 1.0
            ),
            "executed_actions": [
                {
                    "operation": edit.operation.name,
                    "target": list(edit.target),
                }
                for edit in executed.action
            ],
            "executed_decoded_reward": executed_reward_details,
            "mean_decoded_diagnostics": mean_decoded_diagnostics,
        }
        epoch_indices = epoch_transition_indices(epoch)
        epoch_finished = epoch_position + 1 == len(epoch_indices)
        logs.append(record)
        if global_step % args.log_every == 0 or epoch_finished:
            print(
                f"epoch={epoch:03d} step={epoch_position:03d} "
                f"transition={record['transition_id']} "
                f"world={record['world_intervention_loss']:.4f} "
                f"forecast={record['observed_forecast_loss']:.4f} "
                f"synthetic={record['synthetic_intervention_loss']:.4f} "
                f"grpo={record['grpo_loss']:.4f} "
                f"plan={record['action_plan_consistency_loss']:.4f} "
                f"decoder_plan={record['action_decoder_consistency_loss']:.4f} "
                f"decoder_value={record['action_value_decoder_consistency_loss']:.4f} "
                f"adv={record['action_transition_advantage_loss']:.4f} "
                f"obs_adv={record['action_observable_advantage_loss']:.4f} "
                f"a_scale={record['action_residual_scale']:.4f} "
                f"reward_mean={record['mean_reward']:.4f} "
                f"reward_max={record['max_reward']:.4f} "
                f"actions={record['executed_action_count']}"
            )
        if profile_timing:
            print(
                "timing="
                f"setup={profile_setup-profile_start:.3f} "
                f"policy_encode={profile_policy-profile_setup:.3f} "
                f"rollout_sample={profile_rollout_sample-profile_policy:.3f} "
                f"rollout_reward={profile_rollout-profile_rollout_sample:.3f} "
                f"train_graph_encode={profile_world_encode-profile_rollout:.3f} "
                f"action_build={profile_action-profile_world_encode:.3f} "
                f"pair_supervision={profile_supervision-profile_action:.3f} "
                f"forecast={profile_forecast-profile_supervision:.3f} "
                f"target_encode={profile_target-profile_forecast:.3f} "
                f"loss={profile_loss-profile_target:.3f} "
                f"world_backward={profile_world-profile_loss:.3f} "
                f"controller={profile_grpo-profile_world:.3f} "
                f"commit={profile_commit-profile_grpo:.3f} "
                f"total={profile_commit-profile_start:.3f}"
            )

        validation_due = (
            epoch_finished
            and args.val_every > 0





            and (
                args.select_passive_warmup_checkpoint
                or epoch + 1 > args.action_conditioning_warmup_epochs
            )
            and (
                (epoch + 1) % args.val_every == 0
                or epoch + 1 == args.epochs
            )
        )
        if validation_due:
            profile_validation_start = profile_mark()
            torch.save(checkpoint_payload(epoch), candidate_checkpoint_path)
            print(
                f"validation_start epoch={epoch + 1:03d} "
                f"split=val history_burn_in={args.validation_history_burn_in}"
            )
            validation_command = [
                sys.executable,
                str(ROOT / "gwm" / "training" / "evaluate_action_rl.py"),
                "--checkpoint",
                str(candidate_checkpoint_path),
                "--dataset",
                dataset_name,
                "--split",
                "val",
                "--device",
                str(args.validation_device or args.device),
                "--results",
                str(validation_result_path),
            ]





            if args.task in {"topology", "node_change", "node_state"}:
                validation_command.append("--skip_controller_diagnostics")
            if args.validation_transitions:
                validation_command.extend(
                    ["--max_eval_transitions", str(args.validation_transitions)]
                )
            if args.validation_history_burn_in:
                validation_command.extend(
                    [
                        "--history_burn_in",
                        str(args.validation_history_burn_in),
                    ]
                )




            if (
                args.task == "topology"
                and args.controller_topology_count_weight > 0.0
            ):
                validation_command.extend(
                    ["--topology_decode_policy", "controller_count_topk"]
                )
                if args.topology_history_prior:
                    validation_command.append("--topology_history_prior")
            elif args.task == "topology" and args.topology_count_weight > 0.0:
                validation_command.extend(
                    ["--topology_decode_policy", "latent_count_topk"]
                )
            if args.persistent_validation_worker:
                if validation_worker is None:
                    validation_worker = _PersistentValidationWorker()
                completed_stderr = validation_worker.evaluate(
                    validation_command[2:]
                )
            elif args.inprocess_validation:





                from contextlib import redirect_stderr, redirect_stdout
                import io
                from gwm.training.evaluate_action_rl import main as evaluate_main

                validation_stdout = io.StringIO()
                validation_stderr = io.StringIO()
                with (
                    _preserve_validation_process_state(device),
                    redirect_stdout(validation_stdout),
                    redirect_stderr(validation_stderr),
                ):
                    evaluate_main(validation_command[2:])
                completed_stderr = validation_stderr.getvalue()
            else:
                completed = subprocess.run(
                    validation_command,
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                completed_stderr = completed.stderr
            validation_payload = json.loads(validation_result_path.read_text())
            profile_validation_end = profile_mark()
            validation_score = _validation_selection_score(
                args.task,
                validation_payload,
                node_state_metric=args.node_state_selection_metric,
                node_change_metric=args.node_change_selection_metric,
            )
            improved = (
                best_validation_score is None
                or validation_score > best_validation_score + 1e-12
            )
            if improved:
                best_validation_score = validation_score
                best_epoch = epoch
                torch.save(checkpoint_payload(epoch), checkpoint_path)
            validation_record = {
                "epoch": int(epoch + 1),
                "selection_score": float(validation_score),
                "improved": bool(improved),
                "best_epoch": None if best_epoch is None else int(best_epoch + 1),
                "best_score": best_validation_score,
                "metric_source": {
                    "topology": (
                        "mean(addition F1, deletion F1)"
                    ),
                    "node_change": (
                        "validation-calibrated "
                        + {
                            "auprc": "AUPRC",
                            "f1": "F1",
                            "harmonic": "harmonic mean(AUPRC, F1)",
                        }[args.node_change_selection_metric]
                    ),
                    "node_state": (
                        f"negative changed-node {args.node_state_selection_metric.upper()}"
                    ),
                    "node_property": "validation raw future-latent GWM NDCG@10",
                }[args.task],
            }
            validation_history.append(validation_record)
            print(
                f"validation epoch={epoch + 1:03d} "
                f"score={validation_score:.6f} improved={improved} "
                f"best_epoch={validation_record['best_epoch']}"
            )
            if profile_timing:
                print(
                    f"timing_validation={profile_validation_end-profile_validation_start:.3f} "
                    "includes=checkpoint_save+process_start+checkpoint_load+replay+metrics"
                )
            if completed_stderr.strip():
                print(completed_stderr.strip())
            if (
                args.patience > 0
                and best_epoch is not None
                and epoch - best_epoch >= args.patience
            ):
                stopped_early = True
                print(
                    f"early_stop epoch={epoch + 1} patience={args.patience} "
                    f"best_epoch={best_epoch + 1}"
                )
                break

    if validation_worker is not None:
        validation_worker.close()

    if best_epoch is None:


        final_epoch = logs[-1]["epoch"]
        torch.save(checkpoint_payload(int(final_epoch)), checkpoint_path)
    result = {
        "experiment": "action_aware_gwm_grpo_multiedit_latent_decoded",
        "task": args.task,
        "dataset": dataset_name,
        "processed": str(processed.relative_to(ROOT)),
        "action_source": "policy_sampled_variable_cardinality_edit_sequence_with_stop",
        "action_dynamics_target": (
            "synthetic_intervention_and_observed_one_step_forecast"
            if args.lambda_synthetic_intervention > 0.0
            and args.lambda_forecast > 0.0
            else (
                "synthetic_intervention_apply_a_to_g_t"
                if args.lambda_synthetic_intervention > 0.0
                else "observed_one_step_forecast_g_t_a_t_to_g_next"
            )
        ),
        "future_information_use": (
            "released_g_next_is_stop_gradient_training_target_and_reward;"
            "never_action_or_inference_input"
        ),
        "history_action_selection": "future_free_greedy_policy",
        "inference_stop_threshold": args.inference_stop_threshold,
        "property_mode": property_mode if args.task == "node_property" else None,
        "property_primary_readout": (
            args.property_primary_readout if args.task == "node_property" else None
        ),
        "property_loss_type": (
            args.property_loss_type if args.task == "node_property" else None
        ),
        "property_history_input": property_history_input,
        "target_encoder_momentum": args.target_encoder_momentum,
        "pretrained_transfer_gate": args.pretrained_transfer_gate,
        "pretrained_transfer_gates": (
            {
                "graph": model.pretrained_graph_transfer_branch.gate_value(),
                "addition": model.topology_add_transfer_adapter.gate_value(),
                "removal": model.topology_remove_transfer_adapter.gate_value(),
            }
            if args.pretrained_transfer_gate
            else None
        ),
        "observable_latent_mode": args.observable_latent_mode,
        "property_decoder_zero_init": args.property_decoder_zero_init,
        "forecast_latent_cosine_weight": args.forecast_latent_cosine_weight,
        "forecast_latent_variance_weight": args.forecast_latent_variance_weight,
        "property_change_threshold_train_q75": property_change_threshold,
        "steps": len(logs),
        "epochs": args.epochs,
        "epochs_completed": int(logs[-1]["epoch"] + 1),
        "val_every": args.val_every,
        "patience": args.patience,
        "stopped_early": stopped_early,
        "best_epoch": None if best_epoch is None else int(best_epoch + 1),
        "best_validation_score": best_validation_score,
        "validation_history": validation_history,
        "available_train_transitions": len(transitions),
        "transitions_per_epoch": len(epoch_transition_indices(0)),
        "transition_sampling": (
            "complete chronological split"
            if len(epoch_transition_indices(0)) == len(transitions)
            else "deterministic rotating contiguous temporal clips"
        ),
        "clip_history_burn_in": args.clip_history_burn_in,
        "clip_history_action_replay": args.clip_history_action_replay,
        "clip_history_scope": (
            "immediately_preceding_observed_source_snapshots_only; "
            "no_future_targets_or_rewards"
            + (
                "; unary proposal-mode replay recomputes Controller actions "
                "from observed source graphs only"
                if args.clip_history_action_replay
                else "; Controller proposal replay disabled"
            )
        ),
        "dynamic_group_history_scope": (
            "full_observed_graph_edit_type_prefix; completed transitions only; "
            "rotating-clip gaps reconstructed from the observed prefix"
        ),
        "structure_reward_history_scope": "full_observed_transition_prefix",
        "rollout_budget": args.rollout_budget,
        "grpo_update_interval": args.grpo_update_interval,
        "world_intervention_rollouts": args.world_intervention_rollouts,
        "world_warmup_epochs": args.world_warmup_epochs,
        "controller_warmup_epochs": args.controller_warmup_epochs,
        "world_action_lr_scale": args.world_action_lr_scale,
        "action_residual_gate_lr_scale": args.action_residual_gate_lr_scale,
        "freeze_action_residual_gate": args.freeze_action_residual_gate,
        "action_plan_consistency_weight": args.action_plan_consistency_weight,
        "action_plan_positive_weight_cap": args.action_plan_positive_weight_cap,
        "action_decoder_consistency_weight": args.action_decoder_consistency_weight,
        "action_decoder_positive_weight_cap": args.action_decoder_positive_weight_cap,
        "action_value_decoder_consistency_weight": (
            args.action_value_decoder_consistency_weight
        ),
        "action_transition_advantage_weight": args.action_transition_advantage_weight,
        "action_transition_advantage_margin": args.action_transition_advantage_margin,
        "action_observable_advantage_weight": args.action_observable_advantage_weight,
        "action_observable_advantage_margin": args.action_observable_advantage_margin,
        "zero_action_anchor_weight": args.zero_action_anchor_weight,
        "action_residual_only_world_loss": args.action_residual_only_world_loss,
        "bounded_action_residual": args.bounded_action_residual,
        "zero_init_action_adapters": args.zero_init_action_adapters,
        "action_adapter_initial_scale": args.action_adapter_initial_scale,
        "action_residual_max_scale": args.action_residual_max_scale,
        "action_adapter_squash": args.action_adapter_squash,
        "action_adapter_temperature": args.action_adapter_temperature,
        "action_injection_mode": args.action_injection_mode,
        "topology_exact_reward_weight": args.topology_exact_reward_weight,
        "topology_action_reward_weight": args.topology_action_reward_weight,
        "node_action_reward_weight": args.node_action_reward_weight,
        "node_action_value_reward_weight": args.node_action_value_reward_weight,
        "sample_action_magnitude": args.sample_action_magnitude,
        "lambda_observable": args.lambda_observable,
        "lambda_synthetic_intervention": args.lambda_synthetic_intervention,
        "synthetic_intervention_action_only": args.synthetic_intervention_action_only,
        "synthetic_action_delta_scale": args.synthetic_action_delta_scale,
        "action_full_value_conditioning": args.action_full_value_conditioning,
        "action_group_protocol": (
            "joint_native_plus_cross_task_edits"
            if args.joint_action_groups
            else "task_specific"
        ),
        "native_action_groups": [
            operation.name for operation in TASK_OPERATIONS[args.task]
        ],
        "enabled_joint_action_groups": [
            operation.name
            for operation in _action_groups_for_step(
                args.task,
                joint_action_groups=args.joint_action_groups,
                global_step=0,
                interval=args.joint_action_group_interval,
            )[0]
        ],
        "joint_action_groups": args.joint_action_groups,
        "joint_action_group_interval": args.joint_action_group_interval,

        "synthetic_topology_actions": args.joint_action_groups,
        "synthetic_topology_interval": args.joint_action_group_interval,
        "lambda_forecast": args.lambda_forecast,
        "forecast_state_loss_mode": args.forecast_state_loss_mode,
        "forecast_state_loss_type": args.forecast_state_loss_type,
        "forecast_all_state_weight": args.forecast_all_state_weight,
        "forecast_current_state_weight": args.forecast_current_state_weight,
        "state_prediction_mode": args.state_prediction_mode,
        "node_change_selection_metric": args.node_change_selection_metric,
        "state_decoder_zero_init": args.state_decoder_zero_init,
        "controller_state_alignment_weight": args.controller_state_alignment_weight,
        "controller_state_alignment_magnitude_weight": (
            args.controller_state_alignment_magnitude_weight
        ),
        "action_conditioning_warmup_epochs": args.action_conditioning_warmup_epochs,
        "select_passive_warmup_checkpoint": args.select_passive_warmup_checkpoint,
        "soft_node_action_conditioning": args.soft_node_action_conditioning,
        "soft_node_action_mode": args.soft_node_action_mode,
        "soft_node_action_probability_calibration": (
            args.soft_node_action_probability_calibration
        ),
        "soft_node_action_prior_smoothing": args.soft_node_action_prior_smoothing,
        "soft_node_action_topk": args.soft_node_action_topk,
        "soft_node_action_count_support": args.soft_node_action_count_support,
        "soft_edge_action_conditioning": args.soft_edge_action_conditioning,
        "soft_edge_action_topk": args.soft_edge_action_topk,
        "soft_edge_action_count_support": args.soft_edge_action_count_support,
        "confidence_gated_soft_action": args.confidence_gated_soft_action,
        "soft_node_magnitude_conditioning": args.soft_node_magnitude_conditioning,
        "change_gated_action_value": args.change_gated_action_value,
        "controller_magnitude_supervised_weight": args.controller_magnitude_supervised_weight,
        "controller_magnitude_loss": args.controller_magnitude_loss,
        "controller_node_count_weight": args.controller_node_count_weight,
        "lambda_action_locality": args.lambda_action_locality,
        "action_locality_margin": args.action_locality_margin,
        "reward_source": "observable_decoders_of_predicted_future_latent",
        "min_actions": args.min_actions,
        "max_actions": args.max_actions,
        "model_parameters": count_parameters(model),
        "action_encoder_parameters": count_parameters(action_encoder),
        "controller_parameters": count_parameters(controller),
        "history": logs,
        "group_sampler": group_sampler.state_dict(),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    save_json(result, result_path)
    print(f"saved_checkpoint={checkpoint_path.relative_to(ROOT)}")
    print(f"saved_results={result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
