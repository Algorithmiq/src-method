"""Utility functions for the different SRC algorithms."""

from __future__ import annotations

from ._backend import (
    NullEvent,
    NullStream,
    current_stream,
    default_rng,
    device_memory,
    device_pool_bytes,
    device_pool_limit,
    gaussian_sketch,
    get_xp,
    host_memory_available,
    is_host,
    new_stream,
    pinned_empty,
    sketch_dtype,
    to_device_async,
    to_host_async,
    to_numpy,
)
from .linalg import truncated_qr

__all__ = [
    "NullEvent",
    "NullStream",
    "current_stream",
    "default_rng",
    "device_memory",
    "device_pool_bytes",
    "device_pool_limit",
    "gaussian_sketch",
    "get_xp",
    "host_memory_available",
    "is_host",
    "new_stream",
    "pinned_empty",
    "sketch_dtype",
    "to_device_async",
    "to_host_async",
    "to_numpy",
    "truncated_qr",
]
