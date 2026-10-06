"""Utility functions for the different SRC algorithms."""

from __future__ import annotations

from ._backend import default_rng, gaussian_sketch, get_xp, sketch_dtype, to_numpy
from .linalg import truncated_qr

__all__ = [
    "default_rng",
    "gaussian_sketch",
    "get_xp",
    "sketch_dtype",
    "to_numpy",
    "truncated_qr",
]
