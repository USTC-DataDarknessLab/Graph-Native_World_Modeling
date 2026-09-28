
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gwm.training.pretrain_worldgraph import main as _train


DATASETS = {
    "T1": ("trade", "genre", "reddit"),
    "T2": ("trade", "un_vote", "contact", "socialevo"),
    "T3": ("flights", "contact", "enron"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain WorldGraph.")
    parser.add_argument("--task", choices=tuple(DATASETS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    if args.dataset not in DATASETS[args.task]:
        parser.error(f"unknown {args.task} dataset: {args.dataset}")
    output = Path("checkpoints/pretraining") / (
        f"{args.task.lower()}_{args.dataset}.pt"
    )
    sys.argv = [sys.argv[0], "--task", args.task, "--dataset", args.dataset,
                "--device", args.device, "--seed", str(args.seed),
                "--checkpoint", str(output)]
    _train()


if __name__ == "__main__":
    main()
