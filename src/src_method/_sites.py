"""Lazy, per-site access to the cores of a stack.

A core is read from its source (a NumPy array, an `np.memmap`, a zarr or HDF5
dataset) only when the sweep reaches its site, and a background thread can read the
next site while the current one is computed. Shapes come from ``.shape`` alone, so
planning never reads any data.
"""

from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ._tensor_train import known_kind, pad_site, padded_shape
from .utils import NullEvent, current_stream, is_host, pinned_empty, to_device_async

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType, TracebackType

    from numpy.typing import NDArray

    from ._tensor_train import Site

# Byte alignment of each core inside a page-locked staging buffer.
_ALIGN = 256


def padded_shapes(
    layers: Sequence[Sequence[Site]],
) -> list[tuple[tuple[int, ...], ...]]:
    """Return, for every site, the padded ``(l, r, u, d)`` shape of each layer.

    Args:
        layers: A stack in ket form.

    Returns:
        One tuple of shapes per site, without reading any data.
    """
    kinds = [known_kind(layer) for layer in layers]
    last = len(layers[0]) - 1
    return [
        tuple(
            padded_shape(tuple(layer[j].shape), kind, j, last)
            for layer, kind in zip(layers, kinds)
        )
        for j in range(last + 1)
    ]


def site_bytes(layers: Sequence[Sequence[Site]]) -> list[int]:
    """Return, for every site, the bytes of its cores over all layers."""
    return [
        sum(
            prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize for layer in layers
        )
        for j in range(len(layers[0]))
    ]


def _aligned(n_bytes: int) -> int:
    return -(-n_bytes // _ALIGN) * _ALIGN


class SiteSource:
    """Padded device cores of a stack, one site at a time.

    Reading from the source and staging into page-locked memory happen on a
    background thread; the host-to-device copy is issued on the current stream when
    the site is requested, so it is ordered with the kernels that use it.

    Use as a context manager, so that the loader thread stops.

    Args:
        layers: A stack in ket form, each layer a sequence of array-likes with
            ``shape``, ``dtype``, ``ndim`` and ``np.asarray`` support.
        xp: Array module (``numpy`` or ``cupy``).
        depth: How many sites `prefetch` may load ahead (0 disables the thread).
    """

    def __init__(
        self, layers: Sequence[Sequence[Site]], xp: ModuleType, *, depth: int
    ) -> None:
        self._layers = layers
        self._kinds = [known_kind(layer) for layer in layers]
        self._last = len(layers[0]) - 1
        self._xp = xp
        self._pending: dict[int, Future[tuple[list[Any], Any]]] = {}
        self._executor = (
            ThreadPoolExecutor(1, thread_name_prefix="src-site-loader")
            if depth > 0
            else None
        )
        self._buffers: queue.Queue[tuple[Any, Any]] = queue.Queue()
        self.stall_seconds = 0.0
        if not is_host(xp):
            n_bytes = max(
                sum(
                    _aligned(prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize)
                    for layer in layers
                )
                for j in range(self._last + 1)
            )
            for _ in range(depth + 1):
                self._buffers.put((pinned_empty(n_bytes, xp), NullEvent()))

    def __len__(self) -> int:
        """Return the number of sites."""
        return self._last + 1

    def __enter__(self) -> Self:
        """Return the source."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the loader thread."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def prefetch(self, j: int) -> None:
        """Start reading site ``j`` in the background, if there is such a site."""
        if self._executor is None or not 0 <= j <= self._last or j in self._pending:
            return
        self._pending[j] = self._executor.submit(self._stage, j)

    def __getitem__(self, j: int) -> tuple[NDArray, ...]:
        """Return the padded cores of site ``j`` on the device."""
        future = self._pending.pop(j, None)
        if future is None:
            staged, buffer = self._stage(j)
        else:
            start = perf_counter()
            staged, buffer = future.result()
            self.stall_seconds += perf_counter() - start
        if is_host(self._xp):
            return tuple(staged)
        stream = current_stream(self._xp)
        cores = [
            core
            if isinstance(core, self._xp.ndarray)
            else to_device_async(core, self._xp, stream)
            for core in staged
        ]
        # The buffer is refilled only once these copies are done.
        self._buffers.put((buffer, stream.record()))
        return tuple(
            pad_site(core, kind, j, self._last)
            for core, kind in zip(cores, self._kinds)
        )

    def _stage(self, j: int) -> tuple[list[Any], Any]:
        """Read site ``j`` into memory.

        Returns padded arrays on the host backend, and on the GPU unpadded
        page-locked copies (or the device arrays given as input) with their buffer.
        """
        if is_host(self._xp):
            padded = [
                pad_site(_read(layer[j]), kind, j, self._last)
                for layer, kind in zip(self._layers, self._kinds)
            ]
            return padded, None
        buffer, previous = self._buffers.get()
        previous.synchronize()
        staged, offset = [], 0
        for layer in self._layers:
            core = layer[j]
            if not isinstance(core, self._xp.ndarray):
                host = np.asarray(core)
                view = buffer[offset : offset + host.nbytes]
                view = view.view(host.dtype).reshape(host.shape)
                np.copyto(view, host)
                offset += _aligned(host.nbytes)
                core = view
            staged.append(core)
        return staged, buffer


def _read(core: Site) -> NDArray:
    """Bring a core into memory; memmaps are read now, not on first touch."""
    if isinstance(core, np.memmap) or not isinstance(core, np.ndarray):
        return np.array(core)
    return core
