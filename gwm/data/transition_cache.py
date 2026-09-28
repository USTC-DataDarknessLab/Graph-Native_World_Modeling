
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from typing import Any

import torch







DEFAULT_TRANSITION_CACHE_BYTES = 256 * 1024**2


def _record_nbytes(value: Any, seen: set[tuple[str, int]]) -> int:

    if torch.is_tensor(value):
        try:
            identity = (str(value.device), int(value.untyped_storage().data_ptr()))
        except (AttributeError, RuntimeError):
            identity = (str(value.device), id(value))
        if identity in seen:
            return 0
        seen.add(identity)
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, Mapping):
        return sum(_record_nbytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_record_nbytes(item, seen) for item in value)
    return 0


def _fresh_containers(value: Any) -> Any:

    if isinstance(value, dict):
        return {key: _fresh_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fresh_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_fresh_containers(item) for item in value)
    return value


class TransitionRecordCache:

    def __init__(self, max_bytes: int = DEFAULT_TRANSITION_CACHE_BYTES) -> None:
        if int(max_bytes) < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self._records: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: Hashable) -> Any | None:
        cached = self._records.get(key)
        if cached is None:
            return None
        self._records.move_to_end(key)
        return _fresh_containers(cached[0])

    def put(self, key: Hashable, value: Any) -> Any:
        stored = _fresh_containers(value)
        size = _record_nbytes(stored, set())
        previous = self._records.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        if self.max_bytes and size <= self.max_bytes:
            while self._records and self.current_bytes + size > self.max_bytes:
                _old_key, (_old_value, old_size) = self._records.popitem(last=False)
                self.current_bytes -= old_size
            self._records[key] = (stored, size)
            self.current_bytes += size
        return _fresh_containers(value)

    def clear(self) -> None:
        self._records.clear()
        self.current_bytes = 0
