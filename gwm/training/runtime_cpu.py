
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import os
from typing import Any


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configured_cpu_threads(
    environment: Mapping[str, str] | None = None,
    *,
    default: int = 1,
) -> int:

    source = os.environ if environment is None else environment
    raw = source.get("WORLDGRAPH_CPU_THREADS", str(default))
    try:
        threads = int(raw)
    except ValueError as error:
        raise ValueError("WORLDGRAPH_CPU_THREADS must be an integer") from error
    if threads < 1:
        raise ValueError("WORLDGRAPH_CPU_THREADS must be at least one")
    return threads


def configure_cpu_environment(
    environment: MutableMapping[str, str] | None = None,
    *,
    default: int = 1,
    overwrite: bool = False,
) -> int:

    target = os.environ if environment is None else environment
    threads = configured_cpu_threads(target, default=default)
    target["WORLDGRAPH_CPU_THREADS"] = str(threads)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        if overwrite or name not in target:
            target[name] = str(threads)
    if overwrite or "OMP_DYNAMIC" not in target:
        target["OMP_DYNAMIC"] = "FALSE"
    if overwrite or "OMP_WAIT_POLICY" not in target:
        target["OMP_WAIT_POLICY"] = "PASSIVE"
    if overwrite or "MKL_DYNAMIC" not in target:
        target["MKL_DYNAMIC"] = "FALSE"
    return threads


def configure_torch_runtime(torch_module: Any, threads: int) -> None:

    torch_module.set_num_threads(int(threads))
    try:


        torch_module.set_num_interop_threads(1)
    except RuntimeError:

        pass


def subprocess_cpu_environment(
    environment: Mapping[str, str] | None = None,
    *,
    default: int = 1,
) -> dict[str, str]:

    result = dict(os.environ if environment is None else environment)
    configure_cpu_environment(result, default=default, overwrite=True)
    result["PYTHONUNBUFFERED"] = "1"
    return result
