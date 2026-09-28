
from __future__ import annotations

import argparse
from typing import Any


def add_world_model_architecture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph_encoder",
        "--graph_encoder_type",
        dest="graph_encoder_type",
        choices=["sage", "mentor_sgt", "mentor_sgt_gwm"],
        default="mentor_sgt_gwm",
        help=(
            "Graph module f_g: existing GraphSAGE, the mentor-SGT structural "
            "encoder retained as node-wise Z_t, or the "
            "weighted node-wise SGT-GWM instantiation."
        ),
    )
    parser.add_argument(
        "--state_model",
        "--state_model_type",
        dest="state_model_type",
        choices=[
            "gru",
            "history_transformer",
            "mentor_history_transformer",
            "mentor_sgt_gwm_transformer",
        ],
        default="mentor_sgt_gwm_transformer",
        help=(
            "State module f_m: existing node-wise GRU/history Transformer, or "
            "the passive node-wise extension of the mentor history Transformer, or "
            "the node-wise temporal-memory SGT-GWM state module."
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
            "Device-memory budget for reusing deterministic SGT topology and "
            "walk tensors across epochs; zero disables the cache."
        ),
    )
    parser.add_argument(
        "--sgt_include_zero_hop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include projected X_t as a 0-hop candidate in mentor-SGT structural "
            "attention. Disabled by default to preserve the literal supplied equations."
        ),
    )
    parser.add_argument(
        "--sgt_isolated_zero_hop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the projected current node state as the 0-hop Graph Module "
            "output only for nodes without an observed incident edge."
        ),
    )
    parser.add_argument(
        "--sgt_deterministic_walks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a fixed deterministic set of current-snapshot walks in both online and "
            "target encoders; keeps the random-walk branch while removing resampling noise."
        ),
    )
    parser.add_argument(
        "--sgt_attention_dropout",
        type=float,
        default=None,
        help="Defaults to --dropout when omitted.",
    )
    parser.add_argument(
        "--history_window",
        type=int,
        default=8,
        help="Number of past observed graph contexts available to Transformer f_m.",
    )
    parser.add_argument("--history_num_heads", type=int, default=4)
    parser.add_argument(
        "--latent_normalization",
        choices=["none", "layernorm"],
        default="none",
        help="Stateless normalization shared by online Z_t, target Z_(t+1), and predicted latent mean.",
    )
    parser.add_argument("--latent_min_logvar", type=float, default=-8.0)
    parser.add_argument("--latent_max_logvar", type=float, default=5.0)
    parser.add_argument(
        "--latent_transition_mode",
        choices=["absolute", "residual_hidden"],
        default="absolute",
        help=(
            "Predict an absolute mean or a residual correction around H_(t+1). "
            "Both parameterize P(Z_(t+1)|H_(t+1)); absolute preserves prior runs."
        ),
    )


def world_model_architecture_kwargs(args: Any) -> dict[str, Any]:
    return {
        "graph_encoder_type": args.graph_encoder_type,
        "state_model_type": args.state_model_type,
        "sgt_num_hops": args.sgt_num_hops,
        "sgt_num_walks": args.sgt_num_walks,
        "sgt_walk_length": args.sgt_walk_length,
        "sgt_topology_cache_bytes": args.sgt_topology_cache_bytes,
        "sgt_include_zero_hop": args.sgt_include_zero_hop,
        "sgt_isolated_zero_hop": args.sgt_isolated_zero_hop,
        "sgt_deterministic_walks": args.sgt_deterministic_walks,
        "sgt_attention_dropout": args.sgt_attention_dropout,
        "history_window": args.history_window,
        "history_num_heads": args.history_num_heads,
        "latent_normalization": args.latent_normalization,
        "latent_min_logvar": args.latent_min_logvar,
        "latent_max_logvar": args.latent_max_logvar,
        "latent_transition_mode": args.latent_transition_mode,
    }


def architecture_label(args: Any) -> str:
    return f"f_g={args.graph_encoder_type} f_m={args.state_model_type}"
