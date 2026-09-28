
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .action_rl import ActionAwareController
from .benchmark_protocol import load_protocol
from .model import GraphWorldModel
from .models.decoders import NodeChangeDecoder
from .t3_structure import DESCRIPTOR_NAMES, compute_targets








PRETRAINING_FORMAT = "worldgraph-loo-task-aligned-v4"
_COMPATIBLE_PRETRAINING_FORMATS = {
    "worldgraph-loo-contrastive-v2",
    "worldgraph-loo-contrastive-v3",
    PRETRAINING_FORMAT,
}
ACTION_SUMMARY_DIM = 8
TARGET_ARCHITECTURE = {
    "T1": {"latent_dim": 64, "hidden_dim": 64, "action_dim": 32, "policy_dim": 64, "history_window": 8},
    "T2": {"latent_dim": 32, "hidden_dim": 32, "action_dim": 16, "policy_dim": 32, "history_window": 8},
    "T3": {"latent_dim": 64, "hidden_dim": 64, "action_dim": 32, "policy_dim": 64, "history_window": 8},
}


def resolve_target_architecture(task: str, dataset: str) -> dict[str, int]:

    if task not in TARGET_ARCHITECTURE:
        raise ValueError(f"Unknown pretraining task: {task!r}")
    architecture = dict(TARGET_ARCHITECTURE[task])
    key = "".join(character for character in str(dataset).lower() if character.isalnum())



    if task == "T1" and key == "trade":
        architecture.update(
            latent_dim=96, hidden_dim=96, action_dim=48, history_window=12
        )
    return architecture


@dataclass(frozen=True)
class PretrainingSourceSpec:

    key: str
    task: str
    dataset: str
    path: Path
    weighted_edges: bool


def discover_loo_sources(
    target_dataset: str,
    *,
    target_task: str | None = None,
    protocol_path: str | Path | None = None,
    root: str | Path | None = None,
) -> list[PretrainingSourceSpec]:

    project_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )


    def canonical_domain(value: str) -> str:
        key = "".join(character for character in str(value).lower() if character.isalnum())


        for prefix in (
            "flights", "contact", "unvote", "trade", "genre", "reddit",
            "enron", "socialevo",
        ):
            if key.startswith(prefix):
                return prefix
        return key

    target = canonical_domain(target_dataset)
    protocol = load_protocol(protocol_path)
    candidates: dict[str, list[PretrainingSourceSpec]] = {}
    for task, task_spec in protocol["tasks"].items():



        benchmark_datasets = task_spec.get("benchmark_datasets")
        allowed = (
            set(str(value) for value in benchmark_datasets)
            if isinstance(benchmark_datasets, list)
            else set(str(value) for value in task_spec["datasets"])
        )
        audit_only = set(str(value) for value in task_spec.get("audit_only_datasets", []))
        for dataset, dataset_spec in task_spec["datasets"].items():
            if str(dataset) not in allowed or str(dataset) in audit_only:
                continue
            dataset_key = canonical_domain(dataset)
            if dataset_key == target:
                continue
            path = (project_root / dataset_spec["artifact"]).resolve()
            candidates.setdefault(dataset_key, []).append(
                PretrainingSourceSpec(
                    key=f"{task.lower()}_{str(dataset).lower()}",
                    task=str(task),
                    dataset=str(dataset),
                    path=path,
                    weighted_edges=str(dataset_spec.get("edge_input", "binary"))
                    == "weighted",
                )
            )
    task_priority = {
        "T1": ("T1", "T2", "T3"),
        "T2": ("T2", "T3", "T1"),
        "T3": ("T3", "T2", "T1"),
    }
    if target_task is not None and target_task not in task_priority:
        raise ValueError(f"Unknown target task for LOO source selection: {target_task}")
    priority = task_priority.get(target_task or "T1", ("T1", "T2", "T3"))
    sources: list[PretrainingSourceSpec] = []
    for dataset_candidates in candidates.values():
        chosen = min(
            dataset_candidates,
            key=lambda source: priority.index(source.task),
        )
        sources.append(chosen)
    if not sources:
        raise ValueError(f"No LOO sources remain after excluding {target_dataset!r}.")
    missing = [source.path for source in sources if not source.path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing processed LOO source artifacts: {rendered}")
    return sources


def load_snapshot_bundle(path: str | Path) -> dict[str, Any]:

    source = Path(path)
    try:
        payload = torch.load(
            source, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except RuntimeError:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("snapshots"), list):
        raise ValueError(f"Invalid processed snapshot bundle: {source}")
    if len(payload["snapshots"]) < 2:
        raise ValueError(f"Processed source has fewer than two snapshots: {source}")
    return payload


class TemporalPretrainingSource:

    def __init__(self, spec: PretrainingSourceSpec, bundle: dict[str, Any]):
        self.spec = spec
        self.bundle = bundle
        self.snapshots: list[dict[str, Any]] = bundle["snapshots"]
        self.metadata: dict[str, Any] = bundle.get("metadata", {})
        first_x = self.snapshots[0].get("x")
        if not isinstance(first_x, torch.Tensor) or first_x.ndim != 2:
            raise ValueError(f"{spec.key} does not expose dense node features in snapshot['x']")
        self.input_dim = int(first_x.shape[1])
        self._action_summary_cache: dict[int, torch.Tensor] = {}
        self._property_sketch_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._edge_target_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._structure_target_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.indices = {
            split: [
                index
                for index in range(len(self.snapshots) - 1)
                if str(self.snapshots[index + 1].get("split")) == split
            ]
            for split in ("train", "val", "test")
        }
        if not self.indices["train"]:
            raise ValueError(f"{spec.key} has no training transitions")

    def _property_sketch(
        self, snapshot_index: int, *, buckets: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cached = self._property_sketch_cache.get(int(snapshot_index))
        if cached is not None:
            return cached
        snapshot = self.snapshots[int(snapshot_index)]
        num_nodes = int(snapshot["x"].shape[0])
        sketch = torch.zeros((num_nodes, int(buckets)), dtype=torch.float32)
        observed = torch.zeros(num_nodes, dtype=torch.bool)
        node_ids = snapshot.get("property_node_ids")
        flat = snapshot.get("property_target_flat_index")
        values = snapshot.get("property_target_nonzero")
        shape = snapshot.get("property_target_shape")
        if (
            isinstance(node_ids, torch.Tensor)
            and isinstance(flat, torch.Tensor)
            and isinstance(values, torch.Tensor)
            and isinstance(shape, (tuple, list))
            and len(shape) == 2
            and int(shape[1]) > 0
        ):
            node_ids = node_ids.to(dtype=torch.long)
            valid_nodes = node_ids[(node_ids >= 0) & (node_ids < num_nodes)]
            observed.index_fill_(0, valid_nodes, True)
            property_dim = int(shape[1])
            flat = flat.to(dtype=torch.long)
            row = torch.div(flat, property_dim, rounding_mode="floor")
            column = flat.remainder(property_dim)
            valid = (row >= 0) & (row < node_ids.numel())
            row = row[valid]
            column = column[valid]
            value = values.to(dtype=torch.float32)[valid]
            nodes = node_ids.index_select(0, row)
            valid = (nodes >= 0) & (nodes < num_nodes)
            nodes = nodes[valid]
            column = column[valid]
            value = value[valid]
            bucket = column.remainder(int(buckets))
            sign = torch.where(
                torch.div(column, int(buckets), rounding_mode="floor").remainder(2).eq(0),
                torch.ones_like(value),
                -torch.ones_like(value),
            )
            flattened = nodes * int(buckets) + bucket
            sketch.view(-1).scatter_add_(
                0,
                flattened,
                sign * torch.sign(value) * torch.log1p(value.abs()),
            )
            sketch = F.normalize(sketch, dim=-1, eps=1e-8)
        result = (sketch, observed)
        self._property_sketch_cache[int(snapshot_index)] = result
        return result

    def semantic_transition_target(
        self, transition_index: int
    ) -> dict[str, torch.Tensor]:
        current, observed_current = self._property_sketch(int(transition_index))
        following, observed_following = self._property_sketch(int(transition_index) + 1)
        known = observed_current & observed_following
        delta = following - current
        changed = delta.abs().sum(dim=-1).gt(1e-4)
        return {"delta": delta, "changed": changed, "known": known}

    @staticmethod
    def _edge_codes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        edge_index = edge_index.detach().cpu().long()
        return torch.unique(edge_index[0] * int(num_nodes) + edge_index[1])

    def topology_transition_target(
        self, transition_index: int, *, maximum_pairs: int = 512
    ) -> dict[str, torch.Tensor]:

        key = int(transition_index)
        cached = self._edge_target_cache.get(key)
        if cached is not None:
            return cached
        current = self.snapshots[key]
        following = self.snapshots[key + 1]
        num_nodes = int(current["x"].shape[0])
        current_codes = self._edge_codes(current["edge_index"], num_nodes)
        next_codes = self._edge_codes(following["edge_index"], num_nodes)
        add_positive = next_codes[~torch.isin(next_codes, current_codes)]
        remove_positive = current_codes[~torch.isin(current_codes, next_codes)]
        persistent = current_codes[torch.isin(current_codes, next_codes)]
        full_counts = torch.tensor(
            [float(add_positive.numel()), float(remove_positive.numel())],
            dtype=torch.float32,
        )

        generator = torch.Generator(device="cpu")
        seed = sum((position + 1) * ord(char) for position, char in enumerate(self.spec.key))
        generator.manual_seed(seed + 104729 * key)

        def cap(values: torch.Tensor, maximum: int) -> torch.Tensor:
            if values.numel() <= maximum:
                return values
            order = torch.randperm(values.numel(), generator=generator)[:maximum]
            return values.index_select(0, order)

        add_positive = cap(add_positive, int(maximum_pairs) // 2)
        remove_positive = cap(remove_positive, int(maximum_pairs) // 2)
        remove_negative = cap(
            persistent,
            max(int(remove_positive.numel()), min(64, int(persistent.numel()))),
        )
        wanted_add_negatives = max(int(add_positive.numel()), 64)
        add_negative_values: list[torch.Tensor] = []
        occupied = torch.unique(torch.cat((current_codes, add_positive)))
        attempts = 0
        while sum(int(value.numel()) for value in add_negative_values) < wanted_add_negatives:
            attempts += 1
            if attempts > 32:
                break
            proposal = torch.randint(
                0,
                max(num_nodes, 1),
                (2, max(128, 2 * wanted_add_negatives)),
                generator=generator,
            )
            valid = proposal[0].ne(proposal[1])
            codes = proposal[0] * num_nodes + proposal[1]
            codes = torch.unique(codes[valid])
            codes = codes[~torch.isin(codes, occupied)]
            if codes.numel():
                add_negative_values.append(codes)
        add_negative = (
            cap(torch.unique(torch.cat(add_negative_values)), wanted_add_negatives)
            if add_negative_values
            else torch.empty(0, dtype=torch.long)
        )

        def pairs(codes: torch.Tensor) -> torch.Tensor:
            if codes.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long)
            return torch.stack(
                (torch.div(codes, num_nodes, rounding_mode="floor"), codes.remainder(num_nodes))
            )

        result = {
            "add_positive": pairs(add_positive),
            "add_negative": pairs(add_negative),
            "remove_positive": pairs(remove_positive),
            "remove_negative": pairs(remove_negative),
            "counts": full_counts,
        }
        self._edge_target_cache[key] = result
        return result

    def structure_transition_target(self, transition_index: int) -> dict[str, torch.Tensor]:

        key = int(transition_index)
        cached = self._structure_target_cache.get(key)
        if cached is not None:
            return cached
        current = self.snapshots[key]
        following = self.snapshots[key + 1]
        transition: dict[str, Any] = {
            "edge_index_t": current["edge_index"],
            "edge_index_next": following["edge_index"],
        }
        if isinstance(current.get("t3_centres"), torch.Tensor):
            transition["t3_centres"] = current["t3_centres"]
        result = compute_targets(transition)
        self._structure_target_cache[key] = result
        return result

    def history_indices(self, transition_index: int, history_window: int) -> range:
        start = max(0, int(transition_index) - int(history_window) + 1)
        return range(start, int(transition_index) + 1)

    def edge_weight(self, snapshot: dict[str, Any]) -> torch.Tensor | None:
        if not self.spec.weighted_edges:
            return None
        value = snapshot.get("edge_weight")
        return value if isinstance(value, torch.Tensor) else None

    def action_summary(self, transition_index: int) -> torch.Tensor:

        transition_index = int(transition_index)
        cached = self._action_summary_cache.get(transition_index)
        if cached is not None:
            return cached
        if transition_index < 0 or transition_index >= len(self.snapshots) - 1:
            raise IndexError(f"transition_index={transition_index} has no following snapshot")
        current = self.snapshots[transition_index]
        following = self.snapshots[transition_index + 1]
        num_nodes = max(int(current["x"].shape[0]), 1)
        current_edges = max(int(current["edge_index"].shape[1]), 1)

        added = current.get("edge_added")
        removed = current.get("edge_removed")
        if isinstance(added, torch.Tensor) and isinstance(removed, torch.Tensor):
            added_count = int(added.shape[1])
            removed_count = int(removed.shape[1])
        else:



            current_codes = torch.unique(
                current["edge_index"][0].long() * num_nodes
                + current["edge_index"][1].long()
            )
            following_codes = torch.unique(
                following["edge_index"][0].long() * num_nodes
                + following["edge_index"][1].long()
            )
            added_count = int((~torch.isin(following_codes, current_codes)).sum().item())
            removed_count = int((~torch.isin(current_codes, following_codes)).sum().item())
        changed = current.get("node_changed")
        if isinstance(changed, torch.Tensor):
            changed_count = float(changed.to(dtype=torch.float32).sum().item())
        elif current["x"].shape == following["x"].shape:
            changed_count = float(
                (current["x"].to(torch.float32) != following["x"].to(torch.float32))
                .any(dim=-1)
                .sum()
                .item()
            )
        else:
            changed_count = 0.0
        delta = current.get("node_delta")
        if isinstance(delta, torch.Tensor) and delta.numel():
            absolute_delta = delta.to(dtype=torch.float32).abs()
            mean_absolute_delta = float(absolute_delta.mean().item())
            maximum_absolute_delta = float(absolute_delta.amax().item())
        elif current["x"].shape == following["x"].shape:
            absolute_delta = (
                following["x"].to(torch.float32) - current["x"].to(torch.float32)
            ).abs()
            mean_absolute_delta = float(absolute_delta.mean().item())
            maximum_absolute_delta = float(absolute_delta.amax().item())
        else:
            mean_absolute_delta = 0.0
            maximum_absolute_delta = 0.0
        time_t = float(current.get("time_end", current.get("snapshot_id", 0)))
        time_next = float(following.get("time_end", following.get("snapshot_id", 1)))
        next_edges = int(following["edge_index"].shape[1])
        values = torch.tensor(
            [
                torch.log1p(torch.tensor(float(added_count))).item(),
                torch.log1p(torch.tensor(float(removed_count))).item(),
                added_count / current_edges,
                removed_count / current_edges,
                changed_count / num_nodes,
                mean_absolute_delta,
                maximum_absolute_delta,
                torch.sign(torch.tensor(float(next_edges - current_edges))).item()
                * torch.log1p(torch.tensor(abs(float(next_edges - current_edges)))).item(),
            ],
            dtype=torch.float32,
        )
        values = torch.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )



        if time_next < time_t:
            raise ValueError(f"Non-chronological source transition in {self.spec.key}")
        self._action_summary_cache[transition_index] = values
        return values


class _ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(value), dim=-1, eps=1e-8)


class WorldGraphContrastivePretrainer(nn.Module):

    def __init__(
        self,
        input_dims: dict[str, int],
        *,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        policy_dim: int | None = None,
        history_window: int = 8,
        history_num_heads: int = 4,
        dropout: float = 0.1,
        contrastive_dim: int = 64,
        target_momentum: float = 0.99,
        sgt_num_hops: int = 2,
        sgt_num_walks: int = 4,
        sgt_walk_length: int = 3,
        sgt_topology_cache_bytes: int = 0,
        target_task: str | None = None,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("At least one source-domain input dimension is required.")
        self.input_dims = {str(key): int(value) for key, value in input_dims.items()}
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.policy_dim = int(hidden_dim if policy_dim is None else policy_dim)
        self.history_window = int(history_window)
        self.target_momentum = float(target_momentum)
        self.target_task = None if target_task is None else str(target_task)
        if self.target_task not in {None, "T1", "T2", "T3"}:
            raise ValueError(f"Unknown task-aligned pretraining target: {target_task!r}")



        encoder_cache_budget = max(0, int(sgt_topology_cache_bytes) // 2)
        self.world_model = GraphWorldModel(
            input_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            dropout=float(dropout),
            latent_distribution="deterministic",
            observable_latent_mode="mean",
            target_encoder_momentum=float(target_momentum),
            latent_normalization="layernorm",
            graph_encoder_type="mentor_sgt_gwm",
            state_model_type="mentor_sgt_gwm_transformer",
            sgt_num_hops=int(sgt_num_hops),
            sgt_num_walks=int(sgt_num_walks),
            sgt_walk_length=int(sgt_walk_length),
            sgt_topology_cache_bytes=encoder_cache_budget,
            sgt_deterministic_walks=False,
            history_window=self.history_window,
            history_num_heads=int(history_num_heads),
            separate_action_query=True,
            bounded_action_residual=True,
            zero_init_action_adapters=True,
            action_adapter_initial_scale=0.05,
            action_residual_max_scale=0.15,
            action_adapter_squash=True,
            action_adapter_temperature=0.25,
            action_injection_mode="post_norm_residual",
            input_adapter_type="domain_invariant",
        )
        self.action_adapter = nn.Sequential(
            nn.LayerNorm(ACTION_SUMMARY_DIM),
            nn.Linear(ACTION_SUMMARY_DIM, self.action_dim),
            nn.Tanh(),
        )



        self.action_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.latent_dim),
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, ACTION_SUMMARY_DIM),
        )


        with torch.random.fork_rng(devices=[]):
            self.activity_predictor = nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 2),
            )





        with torch.random.fork_rng(devices=[]):
            self.activity_policy_encoder = nn.Sequential(
                nn.Linear(self.latent_dim + self.hidden_dim, self.policy_dim),
                nn.LayerNorm(self.policy_dim),
                nn.GELU(),
            )
            self.activity_policy_head = nn.Linear(self.policy_dim, 2)
        self.node_projector = _ProjectionHead(self.latent_dim, contrastive_dim)
        self.graph_projector = _ProjectionHead(self.latent_dim, contrastive_dim)
        self.state_projector = _ProjectionHead(self.hidden_dim, contrastive_dim)
        self.target_node_projector = copy.deepcopy(self.node_projector).requires_grad_(False)
        self.target_graph_projector = copy.deepcopy(self.graph_projector).requires_grad_(False)



        self.semantic_change_decoder: NodeChangeDecoder | None = None
        self.semantic_delta_decoder: nn.Module | None = None
        self.topology_controller: ActionAwareController | None = None
        self.structure_trunk: nn.Module | None = None
        self.structure_magnitude_head: nn.Linear | None = None
        self.structure_direction_head: nn.Linear | None = None
        with torch.random.fork_rng(devices=[]):
            if self.target_task == "T1":
                self.semantic_change_decoder = NodeChangeDecoder(self.latent_dim)
                self.semantic_delta_decoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.GELU(),
                    nn.Linear(self.latent_dim, 32),
                )
            elif self.target_task == "T2":
                self.topology_controller = ActionAwareController(
                    self.latent_dim,
                    self.hidden_dim,
                    policy_dim=self.policy_dim,
                )
            elif self.target_task == "T3":
                self.structure_trunk = nn.Sequential(
                    nn.Linear(2 * self.latent_dim + self.hidden_dim, self.latent_dim),
                    nn.GELU(),
                )
                self.structure_magnitude_head = nn.Linear(
                    self.latent_dim, len(DESCRIPTOR_NAMES)
                )
                self.structure_direction_head = nn.Linear(
                    self.latent_dim, 3 * len(DESCRIPTOR_NAMES)
                )

    def adapt(self, source_key: str, x: torch.Tensor) -> torch.Tensor:
        del source_key
        assert self.world_model.input_adapter is not None
        return self.world_model.input_adapter(x)

    def adapt_target(self, source_key: str, x: torch.Tensor) -> torch.Tensor:
        del source_key
        assert self.world_model.target_input_adapter is not None
        return self.world_model.target_input_adapter(x)

    def mask_features(
        self,
        x: torch.Tensor,
        probability: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return x
        if probability >= 1.0:
            raise ValueError("feature-mask probability must be smaller than one")
        keep = torch.rand(
            x.shape,
            device=x.device,
            generator=generator,
        ).ge(float(probability))
        return x * keep.to(dtype=x.dtype)

    def encode_history(
        self,
        source: TemporalPretrainingSource,
        transition_index: int,
        *,
        device: torch.device,
        feature_mask_probability: float,
        topology_namespace: str,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        snapshots = source.snapshots
        self.world_model.reset_history()
        first_snapshot = snapshots[next(iter(source.history_indices(transition_index, self.history_window)))]
        num_nodes = int(first_snapshot["x"].shape[0])
        hidden = self.world_model.initial_hidden(num_nodes, device)
        latest_z: torch.Tensor | None = None
        latest_graph: torch.Tensor | None = None
        latest_action: torch.Tensor | None = None
        for snapshot_index in source.history_indices(transition_index, self.history_window):
            snapshot = snapshots[snapshot_index]
            if int(snapshot["x"].shape[0]) != num_nodes:
                raise ValueError(
                    f"{source.spec.key} changes node universe inside one history window"
                )
            x = snapshot["x"].to(device=device, dtype=torch.float32, non_blocking=True)
            x = self.mask_features(
                x,
                feature_mask_probability,
                generator=generator,
            )
            edge_index = snapshot["edge_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            edge_weight = source.edge_weight(snapshot)
            if edge_weight is not None:
                edge_weight = edge_weight.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            encoder_kwargs: dict[str, object] = {
                "edge_weight": edge_weight,
                "topology_cache_key": (
                    source.spec.key,
                    topology_namespace,
                    int(snapshot.get("snapshot_id", snapshot_index)),
                ),
            }
            z_raw = self.world_model.encode_observed_graph(
                x,
                edge_index,
                edge_weight,
                topology_cache_key=encoder_kwargs["topology_cache_key"],
            )
            z = self.world_model._normalize_latent(z_raw)
            graph_context = z.mean(dim=0)




            if snapshot_index == int(transition_index):
                action_summary = torch.zeros(ACTION_SUMMARY_DIM, dtype=torch.float32)
            elif snapshot_index == 0:
                action_summary = torch.zeros(ACTION_SUMMARY_DIM, dtype=torch.float32)
            else:
                action_summary = source.action_summary(snapshot_index - 1)
            action_summary = action_summary.to(device=device)
            action = self.action_adapter(action_summary)
            action_nodes = action.unsqueeze(0).expand(num_nodes, -1)
            hidden = self.world_model.state_model(
                z,
                hidden,
                action=action_nodes,
                graph_context=graph_context,
                commit_history=True,
            )
            latest_z = z
            latest_graph = graph_context
            latest_action = action_nodes
        assert latest_z is not None and latest_graph is not None and latest_action is not None
        latent = self.world_model.latent_predictor(hidden, sample=False)["mu"]
        return {
            "node": latest_z,
            "graph": latest_graph,
            "hidden": hidden,
            "action": latest_action,
            "future": latent,
        }

    def encode_future_target(
        self,
        source: TemporalPretrainingSource,
        transition_index: int,
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        snapshot = source.snapshots[int(transition_index) + 1]
        with torch.no_grad():
            x = snapshot["x"].to(device=device, dtype=torch.float32, non_blocking=True)
            edge_index = snapshot["edge_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            edge_weight = source.edge_weight(snapshot)
            if edge_weight is not None:
                edge_weight = edge_weight.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            target = self.world_model.encode_target(
                x,
                edge_index,
                edge_weight,
                topology_cache_key=(
                    source.spec.key,
                    "target",
                    int(snapshot.get("snapshot_id", transition_index + 1)),
                ),
            )
        return {"node": target, "graph": target.mean(dim=0)}

    @torch.no_grad()
    def update_targets(self) -> None:
        self.world_model.update_target_encoder()
        momentum = self.target_momentum
        for target, online in (
            (self.target_node_projector, self.node_projector),
            (self.target_graph_projector, self.graph_projector),
        ):
            target_parameters = list(target.parameters())
            online_parameters = list(online.parameters())
            torch._foreach_mul_(target_parameters, momentum)
            torch._foreach_add_(
                target_parameters, online_parameters, alpha=1.0 - momentum
            )
    def transferable_state_dict(self) -> dict[str, torch.Tensor]:

        state: dict[str, torch.Tensor] = {}
        if self.world_model.input_adapter is not None:
            for name, value in self.world_model.input_adapter.state_dict().items():
                state[f"input_adapter.{name}"] = value.detach().cpu()
        for component in ("graph_encoder", "state_model", "latent_predictor"):
            module = getattr(self.world_model, component)
            for name, value in module.state_dict().items():
                state[f"{component}.{name}"] = value.detach().cpu()
        return state

    def transferable_activity_controller_state_dict(self) -> dict[str, torch.Tensor]:

        state = {
            f"node_encoder.{name}": value.detach().cpu()
            for name, value in self.activity_policy_encoder.state_dict().items()
        }
        state["node_target_head.weight"] = (
            self.activity_policy_head.weight[0:1].detach().cpu()
        )
        state["node_target_head.bias"] = (
            self.activity_policy_head.bias[0:1].detach().cpu()
        )
        return state

    def transferable_task_state_dict(self) -> dict[str, torch.Tensor]:

        modules: dict[str, nn.Module] = {}
        if self.semantic_change_decoder is not None:
            modules["semantic_change_decoder"] = self.semantic_change_decoder
        if self.semantic_delta_decoder is not None:
            modules["semantic_delta_decoder"] = self.semantic_delta_decoder
        if self.topology_controller is not None:
            modules["topology_controller"] = self.topology_controller
        if self.structure_trunk is not None:
            modules["structure_trunk"] = self.structure_trunk
        if self.structure_magnitude_head is not None:
            modules["structure_magnitude_head"] = self.structure_magnitude_head
        if self.structure_direction_head is not None:
            modules["structure_direction_head"] = self.structure_direction_head
        state = {
            f"{prefix}.{name}": value.detach().cpu()
            for prefix, module in modules.items()
            for name, value in module.state_dict().items()
        }
        if self.target_task == "T1":



            for name, value in self.activity_policy_encoder.state_dict().items():
                state[f"node_add_controller.node_encoder.{name}"] = value.detach().cpu()
                state[f"node_remove_controller.node_encoder.{name}"] = value.detach().cpu()
            for operation, row in (("add", 0), ("remove", 1)):
                state[f"node_{operation}_controller.node_target_head.weight"] = (
                    self.activity_policy_head.weight[row : row + 1].detach().cpu()
                )
                state[f"node_{operation}_controller.node_target_head.bias"] = (
                    self.activity_policy_head.bias[row : row + 1].detach().cpu()
                )
        return state


def symmetric_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("InfoNCE inputs must have identical [items, dim] shapes")
    if left.shape[0] < 2:
        return 1.0 - F.cosine_similarity(left, right, dim=-1).mean()
    logits = left @ right.transpose(0, 1) / float(temperature)
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def graph_queue_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    queue: Iterable[torch.Tensor],
    *,
    temperature: float,
) -> torch.Tensor:

    negatives = list(queue)
    positive = (left * right).sum().reshape(1)
    if not negatives:
        return 1.0 - positive.mean()
    negative_matrix = torch.stack(negatives, dim=0).to(
        device=left.device, dtype=left.dtype
    )
    logits = torch.cat(
        [positive, left @ negative_matrix.transpose(0, 1)], dim=0
    ).unsqueeze(0) / float(temperature)
    return F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=left.device))


def sample_aligned_nodes(
    *values: torch.Tensor,
    maximum: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:

    if not values:
        return tuple()
    node_count = min(int(value.shape[0]) for value in values)
    if node_count < 1:
        raise ValueError("Cannot contrast an empty node set")
    if node_count <= int(maximum):
        indices = torch.arange(node_count, device=values[0].device)
    else:
        indices = torch.randperm(
            node_count, device=values[0].device, generator=generator
        )[: int(maximum)]
    return tuple(value.index_select(0, indices) for value in values)


def load_worldgraph_pretrained_backbone(
    model: GraphWorldModel,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_task: str | None = None,
    expected_dataset: str | None = None,



    backbone_blend: float = 1.0,
    graph_encoder_blend: float | None = None,
    state_model_blend: float | None = None,
    latent_predictor_blend: float | None = None,
    transfer_state_model: bool = True,
    transfer_mode: str = "legacy",
) -> dict[str, Any]:

    if transfer_mode not in {"legacy", "structural"}:
        raise ValueError("transfer_mode must be legacy or structural")
    path = Path(checkpoint).expanduser().resolve()
    component_blends = {
        "graph_encoder": (
            float(backbone_blend)
            if graph_encoder_blend is None
            else float(graph_encoder_blend)
        ),
        "state_model": (
            float(backbone_blend)
            if state_model_blend is None
            else float(state_model_blend)
        ),
        "latent_predictor": (
            float(backbone_blend)
            if latent_predictor_blend is None
            else float(latent_predictor_blend)
        ),
    }
    invalid_blends = {
        name: value
        for name, value in {"backbone": float(backbone_blend), **component_blends}.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid_blends:
        raise ValueError(
            "pretrained blend fractions must lie in [0, 1]: "
            + ", ".join(f"{name}={value}" for name, value in invalid_blends.items())
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("format") not in _COMPATIBLE_PRETRAINING_FORMATS
    ):
        raise ValueError(f"Not a {PRETRAINING_FORMAT} checkpoint: {path}")
    if expected_task is not None and payload.get("target_task") != expected_task:
        raise ValueError(
            f"Pretraining checkpoint targets task={payload.get('target_task')!r}, "
            f"not {expected_task!r}: {path}"
        )
    if expected_dataset is not None:
        def _dataset_alias(value: object) -> str:
            key = "".join(
                character for character in str(value).lower()
                if character.isalnum()
            )
            aliases = {
                "unvote": "unvote",
            }
            return aliases.get(key, key)

        expected_key = _dataset_alias(expected_dataset)
        checkpoint_dataset = str(payload.get("target_dataset", ""))
        checkpoint_key = _dataset_alias(checkpoint_dataset)
        if checkpoint_key != expected_key:
            raise ValueError(
                f"Pretraining checkpoint targets dataset={checkpoint_dataset!r}, "
                f"not {expected_dataset!r}: {path}"
            )
    config = payload.get("config")
    if isinstance(config, dict):
        compatible_settings = {
            "latent_dim": getattr(model, "latent_dim", None),
            "hidden_dim": getattr(model, "hidden_dim", None),
            "action_dim": getattr(model, "action_dim", None),
            "history_window": getattr(model, "history_window", None),
            "history_num_heads": getattr(model, "history_num_heads", None),
        }
        mismatches = {
            name: (config[name], value)
            for name, value in compatible_settings.items()
            if name in config and value is not None and int(config[name]) != int(value)
        }
        if mismatches:
            raise ValueError(
                "Pretraining checkpoint architecture does not match downstream "
                f"WorldGraph: {mismatches}"
            )
    pretrained = payload.get("transferable_state")
    if not isinstance(pretrained, dict):
        raise ValueError(f"Checkpoint has no transferable_state: {path}")

    current = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    if transfer_mode == "structural":





        required_components = {"graph_encoder": 0}
    else:
        required_components = {
            "input_adapter": 0,
            "graph_encoder": 0,
            "latent_predictor": 0,
        }
        if transfer_state_model:
            required_components["state_model"] = 0
    for name, value in pretrained.items():
        target_name = str(name)
        if transfer_mode == "structural":
            if not (
                target_name.startswith("graph_encoder.")
                or target_name.startswith("state_model.")
                or target_name.startswith("latent_predictor.")
            ):
                skipped[target_name] = "target-specific adapter/projector"
                continue
            if target_name in {
                "graph_encoder.input_projection.weight",
                "graph_encoder.input_projection.bias",
            }:
                skipped[target_name] = "target-specific raw feature projection"
                continue
            if target_name.startswith("state_model.action_"):
                skipped[target_name] = "target-specific action projection"
                continue



            if target_name in {
                "state_model.action_reinforcement",
                "state_model.action_residual_gate",
            }:
                skipped[target_name] = "target-specific action gate"
                continue



        if target_name.startswith("state_model.action_"):
            skipped[target_name] = "task-specific action path"
            continue
        if target_name.startswith("state_model.") and not transfer_state_model:
            skipped[target_name] = "downstream action-conditioned state model"
            continue
        if target_name not in current:
            skipped[target_name] = "missing downstream key"
            continue
        if tuple(current[target_name].shape) != tuple(value.shape):
            skipped[target_name] = (
                f"shape {tuple(value.shape)} != {tuple(current[target_name].shape)}"
            )
            continue




        component = target_name.split(".", 1)[0]
        component_blend = component_blends.get(component, float(backbone_blend))
        if (
            component_blend < 1.0
            and component != "input_adapter"
            and torch.is_floating_point(current[target_name])
        ):
            value = (
                (1.0 - component_blend) * current[target_name].detach().cpu()
                + component_blend * value
            )
        loaded[target_name] = value
        if component in required_components:
            required_components[component] += 1
    missing_components = [name for name, count in required_components.items() if count == 0]
    if missing_components:
        raise ValueError(
            "Pretraining checkpoint has no compatible tensors for: "
            + ", ".join(missing_components)
        )
    model.load_state_dict(loaded, strict=False)
    if model.target_graph_encoder is not None:
        model.target_graph_encoder.load_state_dict(model.graph_encoder.state_dict())
        if model.target_input_adapter is not None and model.input_adapter is not None:
            model.target_input_adapter.load_state_dict(model.input_adapter.state_dict())
    return {
        "path": str(path),
        "format": payload["format"],
        "target_task": payload.get("target_task"),
        "target_dataset": payload.get("target_dataset"),
        "loaded_tensors": len(loaded),
        "loaded_by_component": required_components,
        "skipped_tensors": len(skipped),
        "backbone_blend": float(backbone_blend),
        "component_blends": component_blends,
        "transfer_mode": transfer_mode,
        "skipped": skipped,
    }


def load_worldgraph_pretrained_activity_controller(
    controller: nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    blend: float = 1.0,
) -> dict[str, Any] | None:

    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("pretrained controller blend must lie in [0, 1]")





    if float(blend) == 0.0:
        return None
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    pretrained = payload.get("activity_controller_state")
    if not isinstance(pretrained, dict):
        return None
    current = controller.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for name, value in pretrained.items():
        if name not in current:
            skipped[str(name)] = "missing downstream key"
            continue
        if tuple(current[name].shape) != tuple(value.shape):
            skipped[str(name)] = (
                f"shape {tuple(value.shape)} != {tuple(current[name].shape)}"
            )
            continue
        target = value
        if float(blend) < 1.0 and torch.is_floating_point(current[name]):
            target = (
                (1.0 - float(blend)) * current[name].detach().cpu()
                + float(blend) * value
            )
        loaded[str(name)] = target
    required = {"node_encoder.0.weight", "node_target_head.weight"}
    if not required.issubset(loaded):
        raise ValueError(
            "Pretrained activity Controller is missing compatible tensors: "
            + ", ".join(sorted(required - loaded.keys()))
        )
    controller.load_state_dict(loaded, strict=False)
    return {
        "path": str(path),
        "loaded_tensors": len(loaded),
        "blend": float(blend),
        "skipped": skipped,
    }


def load_worldgraph_pretrained_task_module(
    module: nn.Module,
    checkpoint: str | Path,
    *,
    source_prefix: str,
    map_location: str | torch.device = "cpu",
    blend: float = 0.5,
) -> dict[str, Any] | None:

    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("pretrained task blend must lie in [0, 1]")
    if float(blend) == 0.0:
        return None
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    pretrained = payload.get("task_transferable_state")
    if not isinstance(pretrained, dict):
        return None
    prefix = str(source_prefix).rstrip(".") + "."
    current = module.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for source_name, source_value in pretrained.items():
        source_name = str(source_name)
        if not source_name.startswith(prefix):
            continue
        target_name = source_name[len(prefix) :]
        if target_name not in current:
            skipped[target_name] = "missing downstream key"
            continue
        if tuple(current[target_name].shape) != tuple(source_value.shape):
            skipped[target_name] = (
                f"shape {tuple(source_value.shape)} != {tuple(current[target_name].shape)}"
            )
            continue
        value = source_value.to(dtype=current[target_name].dtype)
        if torch.is_floating_point(current[target_name]) and float(blend) < 1.0:
            value = (
                (1.0 - float(blend)) * current[target_name].detach().cpu()
                + float(blend) * value.cpu()
            )
        loaded[target_name] = value
    if not loaded:
        return None
    module.load_state_dict(loaded, strict=False)
    return {
        "path": str(path),
        "source_prefix": str(source_prefix),
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "blend": float(blend),
        "skipped": skipped,
    }


def load_worldgraph_pretrained_t3_head(
    head: nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    blend: float = 0.5,
    magnitude_blend: float | None = None,
    direction_blend: float | None = None,
) -> dict[str, Any] | None:

    effective_magnitude_blend = float(
        blend if magnitude_blend is None else magnitude_blend
    )
    effective_direction_blend = float(
        blend if direction_blend is None else direction_blend
    )
    if not 0.0 <= effective_magnitude_blend <= 1.0:
        raise ValueError("pretrained T3 magnitude blend must lie in [0, 1]")
    if not 0.0 <= effective_direction_blend <= 1.0:
        raise ValueError("pretrained T3 direction blend must lie in [0, 1]")
    if effective_magnitude_blend == 0.0 and effective_direction_blend == 0.0:
        return None
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get("task_transferable_state")
    if not isinstance(state, dict):
        return None
    current = head.state_dict()
    mapping = (
        ("structure_trunk.0.bias", "network.0.bias", effective_magnitude_blend),
        ("structure_magnitude_head.weight", "network.2.weight", effective_magnitude_blend),
        ("structure_magnitude_head.bias", "network.2.bias", effective_magnitude_blend),
        ("structure_direction_head.weight", "direction_network.2.weight", effective_direction_blend),
        ("structure_direction_head.bias", "direction_network.2.bias", effective_direction_blend),
    )
    loaded: dict[str, torch.Tensor] = {}
    for source_name, target_name, component_blend in mapping:
        if component_blend == 0.0:
            continue
        source_value = state.get(source_name)
        target_value = current.get(target_name)
        if not isinstance(source_value, torch.Tensor) or target_value is None:
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            continue
        value = source_value.to(dtype=target_value.dtype)
        if torch.is_floating_point(target_value) and component_blend < 1.0:
            value = (
                (1.0 - component_blend) * target_value.detach().cpu()
                + component_blend * value.cpu()
            )
        loaded[target_name] = value
    source_first = state.get("structure_trunk.0.weight")
    for target_name, component_blend in (
        ("network.0.weight", effective_magnitude_blend),
        ("direction_network.0.weight", effective_direction_blend),
    ):
        if component_blend == 0.0:
            continue
        target_value = current.get(target_name)
        if not isinstance(source_first, torch.Tensor) or target_value is None:
            continue
        if (
            source_first.ndim != 2
            or target_value.ndim != 2
            or source_first.shape[0] != target_value.shape[0]
            or source_first.shape[1] > target_value.shape[1]
        ):
            continue
        value = target_value.detach().cpu().clone()
        width = int(source_first.shape[1])
        value[:, :width] = (
            (1.0 - component_blend) * value[:, :width]
            + component_blend * source_first.to(dtype=value.dtype)
        )
        loaded[target_name] = value
    source_bias = state.get("structure_trunk.0.bias")
    direction_bias = current.get("direction_network.0.bias")
    if (
        effective_direction_blend > 0.0
        and isinstance(source_bias, torch.Tensor)
        and direction_bias is not None
        and tuple(source_bias.shape) == tuple(direction_bias.shape)
    ):
        loaded["direction_network.0.bias"] = (
            (1.0 - effective_direction_blend) * direction_bias.detach().cpu()
            + effective_direction_blend * source_bias.to(dtype=direction_bias.dtype)
        )
    if not loaded:
        return None
    head.load_state_dict(loaded, strict=False)
    return {
        "path": str(path),
        "loaded_tensors": len(loaded),
        "blend": float(blend),
        "magnitude_blend": effective_magnitude_blend,
        "direction_blend": effective_direction_blend,
    }


__all__ = [
    "ACTION_SUMMARY_DIM",
    "PRETRAINING_FORMAT",
    "TARGET_ARCHITECTURE",
    "PretrainingSourceSpec",
    "TemporalPretrainingSource",
    "WorldGraphContrastivePretrainer",
    "discover_loo_sources",
    "graph_queue_info_nce",
    "load_snapshot_bundle",
    "load_worldgraph_pretrained_backbone",
    "load_worldgraph_pretrained_activity_controller",
    "load_worldgraph_pretrained_task_module",
    "load_worldgraph_pretrained_t3_head",
    "resolve_target_architecture",
    "sample_aligned_nodes",
    "symmetric_info_nce",
]
