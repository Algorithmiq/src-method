"""Array-module backend selection for CPU (numpy) and GPU (cupy).

A single resolver returns the appropriate array module, next to a PRNG factory,
host-transfer helpers and the few stream and memory queries the out-of-core sweep
needs. All hot-loop code paths receive an ``xp`` module and call ``xp.linalg.*`` /
``xp.asarray`` directly, so backend selection adds zero per-op overhead. On the
host backend the stream helpers are synchronous no-ops, so the same algorithm code
runs on both.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from numpy.typing import NDArray


def get_xp(device: str) -> ModuleType:
    """Return the array module for the requested device.

    Args:
        device: ``"cpu"`` for numpy or ``"gpu"`` for cupy.

    Returns:
        The numpy or cupy module.

    Raises:
        ValueError: If ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    if device == "cpu":
        return np
    if device == "gpu":
        import cupy  # noqa: PLC0415  (lazy: optional dependency)

        return cupy
    msg = f"Unknown device {device!r}; expected 'cpu' or 'gpu'."
    raise ValueError(msg)


def default_rng(
    seed: int | None,
) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``.

    Always uses NumPy so that the same seed produces identical draws
    regardless of the device, and avoids CuPy ``Generator`` API
    differences (e.g. missing ``.normal()``).
    """
    return np.random.default_rng(seed)


def to_numpy(arr: NDArray) -> np.ndarray:
    """Bring an array onto the host as a numpy array (no-op for numpy)."""
    if isinstance(arr, np.ndarray):
        return arr
    # cupy.ndarray exposes .get(); fall back to np.asarray for other dispatchers.
    get = getattr(arr, "get", None)
    return get() if callable(get) else np.asarray(arr)


class NullEvent:
    """Stand-in for a CUDA event on the host backend: already complete."""

    def synchronize(self) -> None:
        """Return at once: host work is synchronous."""


class NullStream:
    """Stand-in for a CUDA stream on the host backend: work runs synchronously."""

    def record(self) -> NullEvent:
        """Return an event that is already complete."""
        return NullEvent()

    def wait_event(self, event: NullEvent) -> None:
        """Return at once: there is nothing to wait for."""

    def synchronize(self) -> None:
        """Return at once: host work is synchronous."""


def is_host(xp: ModuleType) -> bool:
    """Whether ``xp`` is the host backend (numpy)."""
    return xp is np


def new_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
    """Return a non-blocking stream for transfers, or a `NullStream` on the host."""
    if is_host(xp):
        return NullStream()
    return xp.cuda.Stream(non_blocking=True)


def current_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
    """Return the stream kernels run on, or a `NullStream` on the host."""
    if is_host(xp):
        return NullStream()
    return xp.cuda.get_current_stream()


def pinned_empty(n_bytes: int, xp: ModuleType) -> np.ndarray:
    """Allocate a flat byte buffer, page-locked on the GPU backend.

    Args:
        n_bytes: The size of the buffer.
        xp: Array module (``numpy`` or ``cupy``).

    Returns:
        A ``uint8`` host array of ``n_bytes`` elements.
    """
    if is_host(xp):
        return np.empty(n_bytes, dtype=np.uint8)
    import cupyx  # noqa: PLC0415  (lazy: optional dependency)

    return cupyx.empty_pinned(n_bytes, dtype=np.uint8)


def to_device_async(host: np.ndarray, xp: ModuleType, stream: Any) -> NDArray:  # noqa: ANN401
    """Copy a host array to a new device array on ``stream``.

    On the host backend this is a plain copy, so the result never aliases a staging
    buffer that is about to be reused.

    Args:
        host: The source; page-locked for the copy to be asynchronous.
        xp: Array module (``numpy`` or ``cupy``).
        stream: The stream that performs the copy.

    Returns:
        The new device array.
    """
    if is_host(xp):
        return host.copy()
    device = xp.empty(host.shape, dtype=host.dtype)
    device.set(host, stream=stream)
    return device


def to_host_async(device: NDArray, out: np.ndarray, stream: Any) -> None:  # noqa: ANN401
    """Copy a device array into a host array on ``stream``.

    The copy is complete once an event recorded on ``stream`` afterwards is.

    Args:
        device: The source.
        out: The destination, of the same shape and dtype; page-locked for the
            copy to be asynchronous.
        stream: The stream that performs the copy.
    """
    if isinstance(device, np.ndarray):
        np.copyto(out, device)
        return
    device.get(stream=stream, out=out, blocking=False)


def device_memory(xp: ModuleType) -> tuple[int, int]:
    """Return the device bytes available to the sweep and the device total.

    Available bytes are the free device memory plus the bytes cached, but unused,
    by cupy's default memory pool.

    Args:
        xp: The cupy module.

    Returns:
        ``(available, total)`` in bytes.
    """
    free, total = xp.cuda.runtime.memGetInfo()
    return free + xp.get_default_memory_pool().free_bytes(), total


def device_pool_bytes(xp: ModuleType) -> int:
    """Return the bytes held by cupy's default memory pool, or 0 on the host.

    The pool keeps freed blocks for reuse, so after a sweep this is its high-water
    mark.
    """
    if is_host(xp):
        return 0
    return xp.get_default_memory_pool().total_bytes()


@contextmanager
def device_pool_limit(xp: ModuleType, budget: int) -> Iterator[None]:
    """Cap cupy's default memory pool at ``budget`` bytes beyond its current use.

    An allocation past the cap fails at once instead of when another allocation
    runs out. The previous limit is restored on exit. No-op on the host backend.

    Args:
        xp: Array module (``numpy`` or ``cupy``).
        budget: The bytes the pool may allocate on top of those in use.

    Yields:
        Nothing.
    """
    if is_host(xp):
        yield
        return
    pool = xp.get_default_memory_pool()
    previous = pool.get_limit()
    pool.set_limit(size=pool.used_bytes() + budget)
    try:
        yield
    finally:
        pool.set_limit(size=previous)


def host_memory_available(meminfo: str = "/proc/meminfo") -> int:
    """Return the host memory available for new allocations, in bytes.

    Reads ``MemAvailable`` on Linux and falls back to the free physical pages.

    Args:
        meminfo: The path of the ``meminfo`` file.

    Returns:
        The available bytes.
    """
    path = Path(meminfo)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
