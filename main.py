
from __future__ import annotations

import argparse
import sys

from gwm.training.launcher import main as launch


DATASETS = {
    "T1": ("trade", "genre", "reddit"),
    "T2": ("trade", "un_vote", "contact", "socialevo"),
    "T3": ("flights", "contact", "enron"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate WorldGraph with the paper configuration."
    )
    parser.add_argument("--task", choices=tuple(DATASETS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and print the run without starting training.")
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset not in DATASETS[args.task]:
        parser.error(f"{dataset!r} is not a {args.task} dataset; choose from "
                     + ", ".join(DATASETS[args.task]))
    args.dataset = dataset
    return args


def main() -> None:
    args = parse_args()
    delegated = ["--model", "worldgraph",
                 "--task", args.task, "--dataset", args.dataset,
                 "--device", args.device, "--seed",
                 *(str(seed) for seed in args.seed)]
    if not args.dry_run:
        delegated.append("--execute")
    sys.argv = [sys.argv[0], *delegated]
    launch()


if __name__ == "__main__":
    main()
