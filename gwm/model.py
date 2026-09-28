
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .input_adapter import DomainInvariantInputAdapter
from .transfer import FrozenGraphTransferBranch, GatedResidualTransferAdapter
from .models import (
    EdgeAdditionDecoder,
    EdgeDecoder,
    EdgeDeletionDecoder,
    GraphEncoder,
    HistoryAwareTransformerStateModel,
    LatentPredictor,
    MentorActionNodewiseHistoryTransformerStateModel,
    MentorGWMNodewiseHistoryTransformerStateModel,
    MentorGWMStateAwareGraphTransformerEncoder,
    MentorNodewiseHistoryTransformerStateModel,
    MentorNodewiseStateAwareGraphTransformerEncoder,
    NodeActivityDecoder,
    NodeChangeDecoder,
    NodeOperationCountDecoder,
    NodePropertyDecoder,
    NodeStateDecoder,
    NodeStateModel,
    PairContextEdgeDecoder,
    TopologyOperationCountDecoder,
)


class GraphWorldModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        action_dim: int = 0,
        latent_distribution: str = "gaussian",
        observable_latent_mode: str = "sample",
        node_property_dim: int | None = None,
        node_activity_decoder: bool = False,
        property_decoder_zero_init: bool = False,
        state_decoder_zero_init: bool = False,
        current_state_reconstruction: bool = False,
        state_prediction_mode: str = "ungated",
        topology_transition_heads: bool = False,
        topology_count_decoder: bool = False,
        topology_pair_state_dim: int = 0,
        num_nodes: int | None = None,
        node_id_embedding_dim: int = 0,
        target_encoder_momentum: float = 0.0,
        latent_normalization: str = "none",
        latent_min_logvar: float = -8.0,
        latent_max_logvar: float = 5.0,
        latent_transition_mode: str = "absolute",
        graph_encoder_type: str = "mentor_sgt_gwm",
        state_model_type: str = "mentor_sgt_gwm_transformer",
        sgt_num_hops: int = 2,
        sgt_num_walks: int = 4,
        sgt_walk_length: int = 3,
        sgt_topology_cache_bytes: int = 0,
        sgt_attention_dropout: float | None = None,
        sgt_include_zero_hop: bool = False,
        sgt_isolated_zero_hop: bool = False,
        sgt_deterministic_walks: bool = False,
        history_window: int = 8,
        history_num_heads: int = 4,
        separate_action_query: bool = False,
        bounded_action_residual: bool = False,
        zero_init_action_adapters: bool = False,
        action_adapter_initial_scale: float = 0.10,
        action_residual_max_scale: float = 0.25,
        action_adapter_squash: bool = False,
        action_adapter_temperature: float = 0.25,
        action_injection_mode: str = "full",
        input_adapter_type: str = "none",
        input_adapter_residual_max_scale: float = 0.10,
        action_plan_decoder: bool = False,
        action_plan_decoder_dim: int = 1,
        node_operation_count_decoder: bool = False,
        property_transition_gate_decoder: bool = False,
        node_change_decoder_input: str = "future",
        node_activity_decoder_input: str = "future",
        node_activity_observed_dim: int = 0,
        node_activity_action_dim: int = 0,
        node_activity_history_dim: int = 0,
        node_activity_history_prior: bool = False,
        node_activity_history_prior_remove_only: bool = False,
        node_activity_separate_heads: bool = False,
        node_activity_decoder_activation: str = "relu",
        node_activity_addition_source_dim: int = 0,
        node_change_source_dim: int = 0,
        node_change_causal_dim: int = 0,
        node_change_history_prior: bool = False,
        detach_node_change_input: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim



        self.graph_transfer_adapter: nn.Module | None = None
        self.state_transfer_adapter: nn.Module | None = None
        self.pretrained_graph_transfer_branch: FrozenGraphTransferBranch | None = None
        self.topology_add_transfer_adapter: GatedResidualTransferAdapter | None = None
        self.topology_remove_transfer_adapter: GatedResidualTransferAdapter | None = None
        self.separate_action_query = bool(separate_action_query)
        self.bounded_action_residual = bool(bounded_action_residual)
        self.zero_init_action_adapters = bool(zero_init_action_adapters)
        self.action_adapter_initial_scale = float(action_adapter_initial_scale)
        self.action_residual_max_scale = float(action_residual_max_scale)
        self.action_adapter_squash = bool(action_adapter_squash)
        self.action_adapter_temperature = float(action_adapter_temperature)
        if action_injection_mode not in {
            "full",
            "current_residual",
            "post_norm_residual",
        }:
            raise ValueError(
                "action_injection_mode must be 'full', 'current_residual', "
                "or 'post_norm_residual'."
            )
        self.action_injection_mode = str(action_injection_mode)
        if input_adapter_type not in {
            "none",
            "domain_invariant",
            "residual_domain_invariant",
        }:
            raise ValueError(
                "input_adapter_type must be 'none', 'domain_invariant', or "
                "'residual_domain_invariant'"
            )
        self.input_adapter_type = str(input_adapter_type)
        self.input_adapter_residual_max_scale = float(
            input_adapter_residual_max_scale
        )
        if self.input_adapter_residual_max_scale < 0.0:
            raise ValueError("input_adapter_residual_max_scale must be non-negative")
        if self.input_adapter_type == "domain_invariant":
            self.input_adapter = DomainInvariantInputAdapter(latent_dim)
            self.raw_input_residual = None
        elif self.input_adapter_type == "residual_domain_invariant":



            with torch.random.fork_rng(devices=[]):
                self.input_adapter = DomainInvariantInputAdapter(latent_dim)
            self.raw_input_residual = nn.Linear(input_dim, latent_dim)
            nn.init.zeros_(self.raw_input_residual.weight)
            nn.init.zeros_(self.raw_input_residual.bias)
        else:
            self.input_adapter = None
            self.raw_input_residual = None
        self.input_adapter_residual_logit = (
            nn.Parameter(torch.zeros(()))
            if self.input_adapter_type == "residual_domain_invariant"
            else None
        )
        graph_input_dim = (
            latent_dim
            if self.input_adapter_type in {"domain_invariant", "residual_domain_invariant"}
            else input_dim
        )




        self.action_plan_decoder_enabled = bool(action_plan_decoder)
        self.action_plan_decoder_dim = int(action_plan_decoder_dim)
        self.node_operation_count_decoder_enabled = bool(node_operation_count_decoder)
        if self.action_plan_decoder_dim < 1:
            raise ValueError("action_plan_decoder_dim must be positive.")
        self.latent_distribution = latent_distribution
        if observable_latent_mode not in {"sample", "mean"}:
            raise ValueError("observable_latent_mode must be sample or mean")






        self.observable_latent_mode = observable_latent_mode
        self.node_property_dim = node_property_dim
        self.node_activity_decoder_enabled = bool(node_activity_decoder)
        self.property_decoder_zero_init = bool(property_decoder_zero_init)
        self.state_decoder_zero_init = bool(state_decoder_zero_init)



        self.current_state_reconstruction = bool(current_state_reconstruction)
        self.topology_transition_heads = bool(topology_transition_heads)
        self.topology_count_decoder_enabled = bool(topology_count_decoder)
        self.topology_pair_state_dim = int(topology_pair_state_dim)
        if self.topology_pair_state_dim < 0:
            raise ValueError("topology_pair_state_dim must be non-negative.")
        if self.topology_pair_state_dim and not self.topology_transition_heads:
            raise ValueError(
                "topology_pair_state_dim requires topology_transition_heads=True."
            )
        if self.topology_count_decoder_enabled and not self.topology_transition_heads:
            raise ValueError("topology_count_decoder requires topology_transition_heads=True.")
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.node_id_embedding_dim = int(node_id_embedding_dim)
        self.target_encoder_momentum = float(target_encoder_momentum)
        if not 0.0 <= self.target_encoder_momentum < 1.0:
            raise ValueError("target_encoder_momentum must be in [0, 1).")
        if latent_normalization not in {"none", "layernorm"}:
            raise ValueError("latent_normalization must be none or layernorm")
        if latent_min_logvar >= latent_max_logvar:
            raise ValueError("latent_min_logvar must be smaller than latent_max_logvar")
        self.latent_normalization = latent_normalization
        self.latent_min_logvar = float(latent_min_logvar)
        self.latent_max_logvar = float(latent_max_logvar)
        if latent_transition_mode not in {"absolute", "residual_hidden"}:
            raise ValueError("latent_transition_mode must be absolute or residual_hidden")
        self.latent_transition_mode = latent_transition_mode
        if state_prediction_mode not in {"ungated", "gated"}:
            raise ValueError("state_prediction_mode must be ungated or gated")
        self.state_prediction_mode = state_prediction_mode
        if node_change_decoder_input not in {
            "future",
            "latent_delta",
            "latent_abs_delta",
            "future_concat_delta",
            "future_concat_abs_delta",
        }:
            raise ValueError(
                "node_change_decoder_input must be future, latent_delta, "
                "latent_abs_delta, future_concat_delta, or future_concat_abs_delta"
            )
        self.node_change_decoder_input = node_change_decoder_input
        if node_activity_decoder_input not in {
            "future",
            "latent_delta",
            "latent_abs_delta",
            "future_concat_delta",
            "future_concat_abs_delta",
            "future_concat_observed",
            "future_concat_delta_observed",
            "future_concat_hidden_observed",
        }:
            raise ValueError(
                "node_activity_decoder_input must be future, latent_delta, "
                "latent_abs_delta, future_concat_delta, or "
                "future_concat_abs_delta, future_concat_observed, or "
                "future_concat_delta_observed, or future_concat_hidden_observed"
            )





        self.node_activity_decoder_input = node_activity_decoder_input
        self.node_activity_observed_dim = int(node_activity_observed_dim)
        self.node_activity_action_dim = int(node_activity_action_dim)
        self.node_activity_history_dim = int(node_activity_history_dim)
        self.node_activity_history_prior = bool(node_activity_history_prior)
        self.node_activity_history_prior_remove_only = bool(
            node_activity_history_prior_remove_only
        )
        self.node_activity_addition_source_dim = int(node_activity_addition_source_dim)
        self.node_change_source_dim = int(node_change_source_dim)
        self.node_change_causal_dim = int(node_change_causal_dim)
        self.node_change_history_prior = bool(node_change_history_prior)
        self.detach_node_change_input = bool(detach_node_change_input)
        self.node_activity_separate_heads = bool(node_activity_separate_heads)
        if (
            self.node_activity_observed_dim < 0
            or self.node_activity_action_dim < 0
            or self.node_activity_history_dim < 0
        ):
            raise ValueError(
                "node-activity context dimensions must be non-negative"
            )
        if self.node_activity_history_dim > self.node_activity_observed_dim:
            raise ValueError(
                "node_activity_history_dim cannot exceed node_activity_observed_dim"
            )
        if self.node_activity_addition_source_dim < 0:
            raise ValueError("node_activity_addition_source_dim must be non-negative")
        if self.node_change_source_dim < 0:
            raise ValueError("node_change_source_dim must be non-negative")
        if self.node_change_causal_dim < 0:
            raise ValueError("node_change_causal_dim must be non-negative")
        if self.node_change_causal_dim > self.input_dim:
            raise ValueError("node_change_causal_dim cannot exceed input_dim")
        if (
            self.node_activity_decoder_input
            in {"future_concat_observed", "future_concat_delta_observed"}
            and self.node_activity_observed_dim < 1
        ):
            raise ValueError(
                "observed node-activity decoder inputs require "
                "node_activity_observed_dim >= 1"
            )
        graph_aliases = {
            "sage": "sage",
            "graphsage": "sage",
            "mentor_sgt": "mentor_sgt",
            "mentor_nodewise_sgt": "mentor_sgt",
            "mentor_sgt_gwm": "mentor_sgt_gwm",
        }
        state_aliases = {
            "gru": "gru",
            "history_transformer": "history_transformer",
            "transformer": "history_transformer",
            "mentor_history_transformer": "mentor_history_transformer",
            "mentor_nodewise_history_transformer": "mentor_history_transformer",
            "mentor_action_history_transformer": "mentor_action_history_transformer",
            "mentor_history_action_transformer": "mentor_action_history_transformer",
            "mentor_sgt_gwm_transformer": "mentor_sgt_gwm_transformer",
            "mentor_nodewise_gwm_transformer": "mentor_sgt_gwm_transformer",
        }
        if graph_encoder_type not in graph_aliases:
            raise ValueError("graph_encoder_type must be sage, mentor_sgt, or mentor_sgt_gwm")
        if state_model_type not in state_aliases:
            raise ValueError(
                "state_model_type must be gru, history_transformer, mentor_history_transformer, "
                "mentor_action_history_transformer, or mentor_sgt_gwm_transformer"
            )
        self.graph_encoder_type = graph_aliases[graph_encoder_type]
        self.state_model_type = state_aliases[state_model_type]
        self.sgt_num_hops = int(sgt_num_hops)
        self.sgt_num_walks = int(sgt_num_walks)
        self.sgt_walk_length = int(sgt_walk_length)
        self.sgt_topology_cache_bytes = int(sgt_topology_cache_bytes)
        self.sgt_include_zero_hop = bool(sgt_include_zero_hop)
        self.sgt_isolated_zero_hop = bool(sgt_isolated_zero_hop)
        self.sgt_deterministic_walks = bool(sgt_deterministic_walks)
        self.history_window = int(history_window)
        self.history_num_heads = int(history_num_heads)
        if self.graph_encoder_type == "sage":
            self.graph_encoder = GraphEncoder(
                graph_input_dim,
                latent_dim,
                dropout=dropout,
                num_nodes=self.num_nodes,
                node_id_embedding_dim=self.node_id_embedding_dim,
            )
        elif self.graph_encoder_type == "mentor_sgt":
            if self.node_id_embedding_dim:
                raise ValueError(
                    "mentor_sgt follows the supplied structural equations and does not add node-ID embeddings."
                )
            self.graph_encoder = MentorNodewiseStateAwareGraphTransformerEncoder(
                graph_input_dim,
                latent_dim,
                max_hops=self.sgt_num_hops,
                num_walks=self.sgt_num_walks,
                walk_length=self.sgt_walk_length,
                dropout=dropout,
                include_zero_hop=self.sgt_include_zero_hop,
                isolated_zero_hop=self.sgt_isolated_zero_hop,
                deterministic_walks=self.sgt_deterministic_walks,
                topology_cache_max_bytes=self.sgt_topology_cache_bytes,
            )
        else:
            if self.node_id_embedding_dim:
                raise ValueError(
                    "mentor_sgt_gwm keeps the mentor structural encoder and does not add node-ID embeddings."
                )
            self.graph_encoder = MentorGWMStateAwareGraphTransformerEncoder(
                graph_input_dim,
                latent_dim,
                max_hops=self.sgt_num_hops,
                num_walks=self.sgt_num_walks,
                walk_length=self.sgt_walk_length,
                dropout=dropout,
                include_zero_hop=self.sgt_include_zero_hop,
                isolated_zero_hop=self.sgt_isolated_zero_hop,
                deterministic_walks=self.sgt_deterministic_walks,
                topology_cache_max_bytes=self.sgt_topology_cache_bytes,
            )





        self.target_graph_encoder: nn.Module | None
        if self.target_encoder_momentum > 0.0:
            self.target_graph_encoder = copy.deepcopy(self.graph_encoder)
            self.target_graph_encoder.requires_grad_(False)
            self.target_input_adapter = (
                copy.deepcopy(self.input_adapter).requires_grad_(False)
                if self.input_adapter is not None
                else None
            )
        else:
            self.target_graph_encoder = None
            self.target_input_adapter = None
        if self.state_model_type == "gru":
            self.state_model = NodeStateModel(latent_dim, hidden_dim, action_dim=action_dim)
        elif self.state_model_type == "history_transformer":
            self.state_model = HistoryAwareTransformerStateModel(
                latent_dim,
                hidden_dim,
                action_dim=action_dim,
                history_window=self.history_window,
                num_heads=self.history_num_heads,
                dropout=dropout,
            )
        elif self.state_model_type == "mentor_history_transformer":
            if action_dim:
                raise ValueError(
                    "mentor_history_transformer is the passive action=None implementation; action_dim must be zero."
                )
            self.state_model = MentorNodewiseHistoryTransformerStateModel(
                latent_dim,
                hidden_dim,
                history_window=self.history_window,
                num_heads=self.history_num_heads,
                dropout=dropout,
            )
        elif self.state_model_type == "mentor_action_history_transformer":
            self.state_model = MentorActionNodewiseHistoryTransformerStateModel(
                latent_dim,
                hidden_dim,
                action_dim=action_dim,
                separate_action_query=self.separate_action_query,
                bounded_action_residual=self.bounded_action_residual,
                zero_init_action_adapters=self.zero_init_action_adapters,
                action_adapter_initial_scale=self.action_adapter_initial_scale,
                action_residual_max_scale=self.action_residual_max_scale,
                action_adapter_squash=self.action_adapter_squash,
                action_adapter_temperature=self.action_adapter_temperature,
                action_injection_mode=self.action_injection_mode,
                history_window=self.history_window,
                num_heads=self.history_num_heads,
                dropout=dropout,
            )
        else:
            self.state_model = MentorGWMNodewiseHistoryTransformerStateModel(
                latent_dim,
                hidden_dim,
                action_dim=action_dim,
                separate_action_query=self.separate_action_query,
                bounded_action_residual=self.bounded_action_residual,
                zero_init_action_adapters=self.zero_init_action_adapters,
                action_adapter_initial_scale=self.action_adapter_initial_scale,
                action_residual_max_scale=self.action_residual_max_scale,
                action_adapter_squash=self.action_adapter_squash,
                action_adapter_temperature=self.action_adapter_temperature,
                action_injection_mode=self.action_injection_mode,
                history_window=self.history_window,
                num_heads=self.history_num_heads,
                dropout=dropout,
            )
        self.latent_predictor = LatentPredictor(
            hidden_dim,
            latent_dim,
            distribution=latent_distribution,
            min_logvar=self.latent_min_logvar,
            max_logvar=self.latent_max_logvar,
            normalization=self.latent_normalization,
            transition_mode=self.latent_transition_mode,
        )


        self.edge_decoder = EdgeDecoder(latent_dim)




        if self.topology_transition_heads:


            self.edge_addition_decoder = (
                PairContextEdgeDecoder(latent_dim, self.topology_pair_state_dim)
                if self.topology_pair_state_dim
                else EdgeAdditionDecoder(latent_dim)
            )
            self.edge_deletion_decoder = (
                PairContextEdgeDecoder(latent_dim, self.topology_pair_state_dim)
                if self.topology_pair_state_dim
                else EdgeDeletionDecoder(latent_dim)
            )
        else:
            self.edge_addition_decoder = None
            self.edge_deletion_decoder = None
        self.topology_count_decoder = (
            TopologyOperationCountDecoder(latent_dim)
            if self.topology_count_decoder_enabled
            else None
        )
        node_change_input_dim = (
            2 * latent_dim
            if self.node_change_decoder_input in {
                "future_concat_delta",
                "future_concat_abs_delta",
            }
            else latent_dim
        )
        self.node_change_decoder = NodeChangeDecoder(
            node_change_input_dim,
            source_dim=self.node_change_source_dim,
            causal_context_dim=self.node_change_causal_dim,
            history_prior=self.node_change_history_prior,
        )
        self.node_state_decoder = NodeStateDecoder(
            latent_dim,
            input_dim,
            zero_init_output=self.state_decoder_zero_init,
        )
        self.current_state_decoder = (
            NodeStateDecoder(latent_dim, input_dim)
            if self.current_state_reconstruction
            else None
        )


        self.node_property_decoder = (
            NodePropertyDecoder(
                latent_dim,
                node_property_dim,
                zero_init_output=self.property_decoder_zero_init,
            )
            if node_property_dim is not None
            else None
        )






        if property_transition_gate_decoder and node_property_dim is not None:



            with torch.random.fork_rng(devices=[]):
                self.property_transition_gate_decoder = NodeChangeDecoder(latent_dim)
        else:
            self.property_transition_gate_decoder = None




        if self.node_activity_decoder_input in {
            "future_concat_delta",
            "future_concat_abs_delta",
        }:
            node_activity_input_dim = 2 * latent_dim
        elif self.node_activity_decoder_input == "future_concat_observed":
            node_activity_input_dim = latent_dim + self.node_activity_observed_dim
        elif self.node_activity_decoder_input == "future_concat_delta_observed":
            node_activity_input_dim = 2 * latent_dim + self.node_activity_observed_dim
        elif self.node_activity_decoder_input == "future_concat_hidden_observed":
            node_activity_input_dim = (
                latent_dim + hidden_dim + self.node_activity_observed_dim
            )
        else:
            node_activity_input_dim = latent_dim
        node_activity_input_dim += self.node_activity_action_dim
        self.node_activity_decoder = (
            NodeActivityDecoder(
                node_activity_input_dim,
                separate_heads=self.node_activity_separate_heads,
                activation=node_activity_decoder_activation,
                addition_source_dim=self.node_activity_addition_source_dim,
                history_context_dim=self.node_activity_history_dim,
                history_prior=self.node_activity_history_prior,
                history_prior_remove_only=self.node_activity_history_prior_remove_only,
            )
            if self.node_activity_decoder_enabled
            else None
        )




        self.action_plan_decoder = (
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, self.action_plan_decoder_dim),
            )
            if self.action_plan_decoder_enabled
            else None
        )




        self.node_operation_count_decoder = (
            NodeOperationCountDecoder(latent_dim, num_operations=3)
            if self.node_operation_count_decoder_enabled
            else None
        )

    def initial_hidden(self, num_nodes: int, device: torch.device | str) -> torch.Tensor:




        reset_history = getattr(self.state_model, "reset_history", None)
        if callable(reset_history):
            reset_history()
        return torch.zeros((num_nodes, self.hidden_dim), device=device)

    def reset_history(self) -> None:

        reset_history = getattr(self.state_model, "reset_history", None)
        if callable(reset_history):
            reset_history()

    def adapt_graph_output(self, latent: torch.Tensor) -> torch.Tensor:

        if self.graph_transfer_adapter is None:
            return latent
        return self.graph_transfer_adapter(latent)

    def encode_observed_graph(
        self,
        x_t: torch.Tensor,
        edge_index_t: torch.Tensor,
        edge_weight_t: torch.Tensor | None = None,
        *,
        apply_dropout: bool | None = None,
        topology_cache_key: object | None = None,
    ) -> torch.Tensor:

        encoder_kwargs: dict[str, object] = {"edge_weight": edge_weight_t}
        if apply_dropout is not None:
            encoder_kwargs["apply_dropout"] = bool(apply_dropout)
        if bool(getattr(self.graph_encoder, "supports_topology_cache", False)):
            encoder_kwargs["topology_cache_key"] = topology_cache_key
        if self.input_adapter_type in {"domain_invariant", "residual_domain_invariant"}:
            if self.input_adapter is None:
                raise RuntimeError("fixed-width input adapter is missing")
            encoder_input = self.input_adapter(x_t)
            if self.raw_input_residual is not None:
                encoder_input = encoder_input + self.raw_input_residual(x_t)
        else:
            encoder_input = x_t
        latent = self.graph_encoder(encoder_input, edge_index_t, **encoder_kwargs)
        if self.pretrained_graph_transfer_branch is not None:
            latent = self.pretrained_graph_transfer_branch(
                latent,
                x_t,
                edge_index_t,
                edge_weight_t,
                topology_cache_key=topology_cache_key,
            )
        return latent

    def encode_target(
        self,
        x_next: torch.Tensor,
        edge_index_next: torch.Tensor,
        edge_weight_next: torch.Tensor | None = None,
        topology_cache_key: object | None = None,
    ) -> torch.Tensor:
        encoder = self.target_graph_encoder or self.graph_encoder




        with torch.inference_mode():
            encoder_kwargs: dict[str, object] = {
                "edge_weight": edge_weight_next,
                "apply_dropout": False,
            }
            if bool(getattr(encoder, "supports_topology_cache", False)):
                encoder_kwargs["topology_cache_key"] = topology_cache_key
            if self.input_adapter_type in {"domain_invariant", "residual_domain_invariant"}:
                if self.target_input_adapter is None:
                    raise RuntimeError("target fixed-width input adapter is missing")
                target_input = self.target_input_adapter(x_next)
                if self.raw_input_residual is not None:
                    target_input = target_input + self.raw_input_residual(x_next)
            else:
                target_input = x_next
            target = encoder(target_input, edge_index_next, **encoder_kwargs)
            if self.pretrained_graph_transfer_branch is not None:
                target = self.pretrained_graph_transfer_branch(
                    target,
                    x_next,
                    edge_index_next,
                    edge_weight_next,
                    topology_cache_key=topology_cache_key,
                )




            target = self._normalize_latent(target).detach()

        return target.clone()

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_normalization == "layernorm":
            return F.layer_norm(latent, (latent.shape[-1],))
        return latent

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_graph_encoder is None:
            return
        momentum = self.target_encoder_momentum
        target_parameters = list(self.target_graph_encoder.parameters())
        online_parameters = list(self.graph_encoder.parameters())



        torch._foreach_mul_(target_parameters, momentum)
        torch._foreach_add_(
            target_parameters, online_parameters, alpha=1.0 - momentum
        )


        target_buffers = list(self.target_graph_encoder.buffers())
        online_buffers = list(self.graph_encoder.buffers())
        if target_buffers:
            torch._foreach_copy_(target_buffers, online_buffers)
        if self.target_input_adapter is not None and self.input_adapter is not None:
            target_parameters = list(self.target_input_adapter.parameters())
            online_parameters = list(self.input_adapter.parameters())
            torch._foreach_mul_(target_parameters, momentum)
            torch._foreach_add_(
                target_parameters, online_parameters, alpha=1.0 - momentum
            )

    def decode_topology_transition(
        self,
        z_next_pred: torch.Tensor,
        *,
        addition_candidate_edges: torch.Tensor | None = None,
        deletion_candidate_edges: torch.Tensor | None = None,
        addition_pair_features: torch.Tensor | None = None,
        deletion_pair_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.topology_transition_heads:
            if addition_candidate_edges is not None or deletion_candidate_edges is not None:
                raise RuntimeError(
                    "topology_transition_heads=True is required for Delta-E decoding."
                )
            return {}
        assert self.edge_addition_decoder is not None
        assert self.edge_deletion_decoder is not None
        outputs: dict[str, torch.Tensor] = {}
        if addition_candidate_edges is not None:
            addition_latent = (
                self.topology_add_transfer_adapter(z_next_pred)
                if self.topology_add_transfer_adapter is not None
                else z_next_pred
            )
            source, destination = addition_candidate_edges
            if addition_latent.ndim == 2:
                source_latent = addition_latent[source]
                destination_latent = addition_latent[destination]
            else:
                source_latent = addition_latent[:, source]
                destination_latent = addition_latent[:, destination]
            pair_features = addition_pair_features
            if pair_features is not None and z_next_pred.ndim == 3 and pair_features.ndim == 2:
                pair_features = pair_features.unsqueeze(0).expand(
                    z_next_pred.shape[0], -1, -1
                )
            outputs["edge_addition_logits"] = self.edge_addition_decoder(
                source_latent, destination_latent, pair_features
            )
        if deletion_candidate_edges is not None:
            deletion_latent = (
                self.topology_remove_transfer_adapter(z_next_pred)
                if self.topology_remove_transfer_adapter is not None
                else z_next_pred
            )
            source, destination = deletion_candidate_edges
            if deletion_latent.ndim == 2:
                source_latent = deletion_latent[source]
                destination_latent = deletion_latent[destination]
            else:
                source_latent = deletion_latent[:, source]
                destination_latent = deletion_latent[:, destination]
            pair_features = deletion_pair_features
            if pair_features is not None and z_next_pred.ndim == 3 and pair_features.ndim == 2:
                pair_features = pair_features.unsqueeze(0).expand(
                    z_next_pred.shape[0], -1, -1
                )
            outputs["edge_deletion_logits"] = self.edge_deletion_decoder(
                source_latent, destination_latent, pair_features
            )
        return outputs

    def forward(
        self,
        x_t: torch.Tensor,
        edge_index_t: torch.Tensor,
        hidden_state: torch.Tensor | None,
        candidate_edges: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        edge_weight_t: torch.Tensor | None = None,
        addition_candidate_edges: torch.Tensor | None = None,
        deletion_candidate_edges: torch.Tensor | None = None,
        addition_pair_features: torch.Tensor | None = None,
        deletion_pair_features: torch.Tensor | None = None,
        decode_node_state: bool = True,
        decode_observables: bool = True,
        decode_node_features: bool = True,
        decode_node_property: bool = True,
        decode_current_property: bool = True,
        decode_property_transition_gate: bool = True,
        node_property_ids: torch.Tensor | None = None,
        current_property_node_ids: torch.Tensor | None = None,
        commit_history: bool = True,
        precomputed_z_t_raw: torch.Tensor | None = None,
        precomputed_z_t_raw_is_adapted: bool = False,
        topology_cache_key: object | None = None,
        node_activity_observed_context: torch.Tensor | None = None,
        node_activity_action_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:







        if precomputed_z_t_raw is None:
            z_t_raw = self.encode_observed_graph(
                x_t,
                edge_index_t,
                edge_weight_t,
                topology_cache_key=topology_cache_key,
            )
        else:
            z_t_raw = precomputed_z_t_raw
        if not precomputed_z_t_raw_is_adapted:
            z_t_raw = self.adapt_graph_output(z_t_raw)
        if node_activity_observed_context is not None:
            if node_activity_observed_context.ndim != 2 or (
                node_activity_observed_context.shape[0] != x_t.shape[0]
            ):
                raise ValueError(
                    "node_activity_observed_context must have shape [num_nodes, context_dim]"
                )
            if (
                self.node_activity_observed_dim
                and node_activity_observed_context.shape[1]
                != self.node_activity_observed_dim
            ):
                raise ValueError(
                    "node_activity_observed_context dimension does not match "
                    "node_activity_observed_dim"
                )
        if node_activity_action_context is not None:
            if node_activity_action_context.ndim != 2 or (
                node_activity_action_context.shape[0] != x_t.shape[0]
            ):
                raise ValueError(
                    "node_activity_action_context must have shape [num_nodes, action_context_dim]"
                )
            if (
                self.node_activity_action_dim
                and node_activity_action_context.shape[1]
                != self.node_activity_action_dim
            ):
                raise ValueError(
                    "node_activity_action_context dimension does not match "
                    "node_activity_action_dim"
                )
        z_t = self._normalize_latent(z_t_raw)



        graph_context_t = z_t.mean(dim=0)
        h_next = self.state_model(
            z_t,
            hidden_state,
            action=action,
            graph_context=graph_context_t,
            commit_history=commit_history,
        )



        h_next_pre_transfer = h_next
        if self.state_transfer_adapter is not None:
            h_next = self.state_transfer_adapter(h_next)
        latent_distribution = self.latent_predictor(h_next)
        z_next_sample = latent_distribution["sample"]
        z_next_pred = (
            latent_distribution["mu"]
            if self.observable_latent_mode == "mean"
            else z_next_sample
        )
        outputs = {
            "z_t_raw": z_t_raw,
            "z_t": z_t,
            "graph_context_t": graph_context_t,
            "hidden_next_pre_transfer": h_next_pre_transfer,
            "hidden_next": h_next,
            "latent_mu_raw": latent_distribution["raw_mu"],
            "latent_delta_mu_raw": latent_distribution["delta_mu_raw"],
            "latent_mu": latent_distribution["mu"],
            "latent_logvar": latent_distribution["logvar"],
            "latent_is_gaussian": self.latent_distribution == "gaussian",
            "latent_sample": z_next_sample,
            "latent_next": z_next_pred,
        }





        if decode_observables and decode_node_state:




            if self.node_change_decoder_input == "latent_delta":
                node_change_latent = z_next_pred - z_t
            elif self.node_change_decoder_input == "latent_abs_delta":
                node_change_latent = (z_next_pred - z_t).abs()
            elif self.node_change_decoder_input == "future_concat_delta":
                node_change_latent = torch.cat(
                    [z_next_pred, z_next_pred - z_t], dim=-1
                )
            elif self.node_change_decoder_input == "future_concat_abs_delta":
                node_change_latent = torch.cat(
                    [z_next_pred, (z_next_pred - z_t).abs()], dim=-1
                )
            else:
                node_change_latent = z_next_pred
            decoder_node_change_latent = (
                node_change_latent.detach()
                if self.detach_node_change_input
                else node_change_latent
            )
            node_change_source_context = (
                x_t.detach() if self.detach_node_change_input else x_t
            )
            node_change_logits = self.node_change_decoder(
                decoder_node_change_latent,
                source_context=(
                    node_change_source_context
                    if self.node_change_source_dim
                    else None
                ),



                causal_context=(
                    x_t[..., -self.node_change_causal_dim :]
                    if self.node_change_causal_dim
                    else None
                ),
            )


            node_change_probability = torch.sigmoid(node_change_logits)
            outputs.update(
                {
                    "node_change_logits": node_change_logits,
                    "node_change_decoder_latent": node_change_latent,
                    "node_change_probability": node_change_probability,
                }
            )






            if decode_node_features:
                node_delta_ungated = self.node_state_decoder(z_next_pred)
                node_delta = (
                    node_change_probability.unsqueeze(-1) * node_delta_ungated
                    if self.state_prediction_mode == "gated"
                    else node_delta_ungated
                )
                outputs.update(
                    {
                        "node_delta_ungated": node_delta_ungated,
                        "node_delta": node_delta,
                        "x_next_pred": x_t + node_delta,
                    }
                )
        if decode_observables and self.topology_count_decoder is not None:
            outputs["topology_count_log1p"] = self.topology_count_decoder(z_next_pred)
        if decode_observables and self.current_state_decoder is not None:



            outputs["x_t_reconstructed"] = self.current_state_decoder(z_t)
        if (
            decode_observables
            and decode_node_property
            and self.node_property_decoder is not None
        ):
            property_latent = (
                z_next_pred
                if node_property_ids is None
                else z_next_pred.index_select(0, node_property_ids)
            )
            outputs["node_property"] = self.node_property_decoder(property_latent)
            if node_property_ids is not None:
                outputs["node_property_node_ids"] = node_property_ids





            if decode_current_property:
                current_property_latent = (
                    z_t
                    if current_property_node_ids is None
                    else z_t.index_select(0, current_property_node_ids)
                )
                outputs["node_property_current"] = self.node_property_decoder(
                    current_property_latent
                )
                if current_property_node_ids is not None:
                    outputs["node_property_current_node_ids"] = (
                        current_property_node_ids
                    )
        if (
            decode_observables
            and decode_property_transition_gate
            and self.property_transition_gate_decoder is not None
        ):
            outputs["node_property_transition_gate_logits"] = (
                self.property_transition_gate_decoder(z_next_pred)
            )
        if decode_observables and self.node_activity_decoder is not None:
            if self.node_activity_decoder_input == "latent_delta":
                node_activity_latent = z_next_pred - z_t
            elif self.node_activity_decoder_input == "latent_abs_delta":
                node_activity_latent = (z_next_pred - z_t).abs()
            elif self.node_activity_decoder_input == "future_concat_delta":
                node_activity_latent = torch.cat(
                    [z_next_pred, z_next_pred - z_t], dim=-1
                )
            elif self.node_activity_decoder_input == "future_concat_abs_delta":
                node_activity_latent = torch.cat(
                    [z_next_pred, (z_next_pred - z_t).abs()], dim=-1
                )
            elif self.node_activity_decoder_input == "future_concat_observed":
                if node_activity_observed_context is None:
                    raise ValueError(
                        "future_concat_observed requires node_activity_observed_context"
                    )
                observed = node_activity_observed_context.to(
                    device=z_next_pred.device, dtype=z_next_pred.dtype
                )
                node_activity_latent = torch.cat([z_next_pred, observed], dim=-1)
            elif self.node_activity_decoder_input == "future_concat_delta_observed":
                if node_activity_observed_context is None:
                    raise ValueError(
                        "future_concat_delta_observed requires "
                        "node_activity_observed_context"
                    )
                observed = node_activity_observed_context.to(
                    device=z_next_pred.device, dtype=z_next_pred.dtype
                )
                node_activity_latent = torch.cat(
                    [z_next_pred, z_next_pred - z_t, observed], dim=-1
                )
            elif self.node_activity_decoder_input == "future_concat_hidden_observed":
                if node_activity_observed_context is None:
                    raise ValueError(
                        "future_concat_hidden_observed requires "
                        "node_activity_observed_context"
                    )
                observed = node_activity_observed_context.to(
                    device=z_next_pred.device, dtype=z_next_pred.dtype
                )
                node_activity_latent = torch.cat(
                    [z_next_pred, h_next, observed], dim=-1
                )
            else:
                node_activity_latent = z_next_pred
            if self.node_activity_action_dim:
                if node_activity_action_context is None:
                    action_context = z_next_pred.new_zeros(
                        (z_next_pred.shape[0], self.node_activity_action_dim)
                    )
                else:
                    action_context = node_activity_action_context.to(
                        device=z_next_pred.device, dtype=z_next_pred.dtype
                    )
                node_activity_latent = torch.cat(
                    [node_activity_latent, action_context], dim=-1
                )
            activity_transition_logits = self.node_activity_decoder(
                node_activity_latent,
                addition_source_context=(
                    x_t if self.node_activity_addition_source_dim else None
                ),
                history_context=(
                    node_activity_observed_context[..., -self.node_activity_history_dim :]
                    if self.node_activity_history_dim
                    and node_activity_observed_context is not None
                    else None
                ),
            )
            outputs["node_activity_logits"] = activity_transition_logits
            outputs["node_activation_logits"] = activity_transition_logits[..., 0]
            outputs["node_deactivation_logits"] = activity_transition_logits[..., 1]
            outputs["node_activity_decoder_latent"] = node_activity_latent
            if self.node_activity_action_dim:
                outputs["node_activity_action_context"] = action_context
        if decode_observables and self.action_plan_decoder is not None:
            action_plan_logits = self.action_plan_decoder(z_next_pred)
            outputs["action_plan_logits"] = (
                action_plan_logits.squeeze(-1)
                if self.action_plan_decoder_dim == 1
                else action_plan_logits
            )
        if decode_observables and self.node_operation_count_decoder is not None:
            outputs["node_operation_count_log1p"] = self.node_operation_count_decoder(
                z_next_pred
            )
        if decode_observables and candidate_edges is not None:
            source, destination = candidate_edges
            if z_next_pred.ndim == 2:
                source_latent = z_next_pred[source]
                destination_latent = z_next_pred[destination]
            else:
                source_latent = z_next_pred[:, source]
                destination_latent = z_next_pred[:, destination]
            outputs["edge_logits"] = self.edge_decoder(
                source_latent, destination_latent
            )
        if decode_observables:
            outputs.update(
                self.decode_topology_transition(
                    z_next_pred,
                    addition_candidate_edges=addition_candidate_edges,
                    deletion_candidate_edges=deletion_candidate_edges,
                    addition_pair_features=addition_pair_features,
                    deletion_pair_features=deletion_pair_features,
                )
            )
        return outputs
