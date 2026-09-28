
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return device


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON")


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def finite_tensor_dict(values: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in values.values())
