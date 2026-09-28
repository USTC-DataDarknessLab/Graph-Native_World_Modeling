
from __future__ import annotations

from dataclasses import dataclass, field

import torch


PAIR_STATE_DIM = 5


@dataclass
class CausalPairState:

    num_nodes: int
    _counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _last_seen: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _run_length: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _cached_codes_cpu: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_values_cpu: torch.Tensor | None = field(default=None, init=False, repr=False)
    _device_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if int(self.num_nodes) < 1:
            raise ValueError("num_nodes must be positive.")
        self.num_nodes = int(self.num_nodes)

    def reset(self) -> None:
        self._counts.clear()
        self._last_seen.clear()
        self._run_length.clear()
        self._invalidate_cache()

    def observed_codes(self) -> torch.Tensor:
        if not self._counts:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(self._counts), dtype=torch.long)

    def _invalidate_cache(self) -> None:
        self._cached_codes_cpu = None
        self._cached_values_cpu = None
        self._device_cache.clear()

    def _codes(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.numel() == 0:
            return edge_index.new_empty((0,), dtype=torch.long)
        return edge_index[0].long() * self.num_nodes + edge_index[1].long()

    def observe(self, edge_index: torch.Tensor, *, time_index: int) -> None:
        if int(time_index) < 0:
            raise ValueError("time_index must be non-negative.")
        codes = torch.unique(self._codes(edge_index).detach().cpu()).tolist()
        for raw_code in codes:
            code = int(raw_code)
            previous_time = self._last_seen.get(code)
            previous_run = self._run_length.get(code, 0)
            self._counts[code] = self._counts.get(code, 0) + 1
            self._last_seen[code] = int(time_index)
            self._run_length[code] = (
                previous_run + 1
                if previous_time is not None and previous_time == int(time_index) - 1
                else 1
            )
        if codes:
            self._invalidate_cache()

    def _sorted_state(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_codes_cpu is None or self._cached_values_cpu is None:
            if not self._counts:
                self._cached_codes_cpu = torch.empty(0, dtype=torch.long)
                self._cached_values_cpu = torch.empty((0, 3), dtype=torch.float32)
            else:
                ordered = sorted(self._counts)
                self._cached_codes_cpu = torch.tensor(ordered, dtype=torch.long)
                self._cached_values_cpu = torch.tensor(
                    [
                        [
                            float(self._counts[code]),
                            float(self._last_seen[code]),
                            float(self._run_length[code]),
                        ]
                        for code in ordered
                    ],
                    dtype=torch.float32,
                )
        key = f"{device.type}:{device.index}:{dtype}"
        cached = self._device_cache.get(key)
        if cached is None:
            cached = (
                self._cached_codes_cpu.to(device=device),
                self._cached_values_cpu.to(device=device, dtype=dtype),
            )
            self._device_cache[key] = cached
        return cached

    def features(
        self,
        edge_index: torch.Tensor,
        *,
        time_index: int,
        currently_present: bool | torch.Tensor,
    ) -> torch.Tensor:
        if int(time_index) < 0:
            raise ValueError("time_index must be non-negative.")
        codes = self._codes(edge_index)
        count = int(codes.numel())
        if isinstance(currently_present, bool):
            present = torch.full(
                (count,), currently_present, device=codes.device, dtype=torch.bool
            )
        else:
            present = currently_present.to(device=codes.device, dtype=torch.bool)
            if present.shape != (count,):
                raise ValueError("currently_present tensor must have one value per pair.")
        if count == 0:
            return torch.empty((0, PAIR_STATE_DIM), device=codes.device, dtype=torch.float32)

        keys, values = self._sorted_state(codes.device, torch.float32)
        history_count = torch.zeros(count, device=codes.device, dtype=torch.float32)
        last_seen = torch.full((count,), -1.0, device=codes.device)
        prior_run = torch.zeros(count, device=codes.device, dtype=torch.float32)
        if keys.numel():
            positions = torch.searchsorted(keys, codes)
            valid = positions < keys.numel()
            matched = valid.clone()
            if bool(valid.any()):
                matched[valid] = keys.index_select(0, positions[valid]).eq(codes[valid])
            if bool(matched.any()):
                selected = values.index_select(0, positions[matched])
                history_count[matched] = selected[:, 0]
                last_seen[matched] = selected[:, 1]
                prior_run[matched] = selected[:, 2]




        observed_count = history_count + present.to(torch.float32)
        ever_observed = observed_count.gt(0)
        age = torch.where(
            ever_observed & ~present,
            (float(time_index) - last_seen).clamp_min(0.0),
            torch.zeros_like(last_seen),
        )
        continues = last_seen.eq(float(time_index - 1))
        run_length = torch.where(
            present,
            torch.where(continues, prior_run + 1.0, torch.ones_like(prior_run)),
            torch.zeros_like(prior_run),
        )
        return torch.stack(
            [
                torch.log1p(observed_count),
                torch.log1p(age),
                torch.log1p(run_length),
                present.to(torch.float32),
                ever_observed.to(torch.float32),
            ],
            dim=-1,
        )


__all__ = ["CausalPairState", "PAIR_STATE_DIM"]
