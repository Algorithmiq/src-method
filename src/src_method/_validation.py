"""Validation helpers for the public `apply` / `compress` arguments.

These centralize the argument checks so `apply.py` and `compress.py` don't
duplicate them. A malformed call therefore fails immediately, before any
array allocation, device transfer, or `infer_kind` dispatch.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# The upper bound of "plausible" boundary ranks worth diagnosing
# specifically -- see _validate_against_interior.
_PERIODIC_BOUNDARY_MIN_NDIM = 4

# The lower bound of "plausible" boundary ranks -- see
# _validate_against_interior.
_MIN_PLAUSIBLE_BOUNDARY_NDIM = 2

# Below this length there's no interior site distinct from both boundaries
# to compare against; see validate_boundary_rank's fallback branch.
_MIN_SITES_FOR_INTERIOR_CHECK = 3

__all__ = [
    "validate_boundary_rank",
    "validate_chi_out",
    "validate_cutoff",
]


def validate_chi_out(chi_out: int) -> None:
    """Reject a non-integer or non-positive output bond dimension.

    `numbers.Integral` is used rather than `isinstance(chi_out, int)` so
    numpy integer types (`np.int64`, etc.), which are a plausible thing for
    a caller to pass, are accepted rather than rejected.

    Args:
        chi_out: The requested output bond dimension.

    Raises:
        ValueError: If `chi_out` is not a positive integer.
    """
    if (
        isinstance(chi_out, bool)
        or not isinstance(chi_out, numbers.Integral)
        or chi_out < 1
    ):
        msg = f"chi_out must be a positive integer, got {chi_out!r}."
        raise ValueError(msg)


def validate_cutoff(cutoff: float) -> None:
    """Reject a cutoff outside the range that is handled sanely.

    Args:
        cutoff: The relative singular-value cutoff.

    Raises:
        ValueError: If `cutoff` is not in `[0.0, 1.0)`.
    """
    if not isinstance(cutoff, numbers.Real) or not 0.0 <= cutoff < 1.0:
        msg = f"cutoff must be in [0.0, 1.0), got {cutoff}."
        raise ValueError(msg)


def validate_boundary_rank(arrays: Sequence[NDArray], *, label: str = "tensor") -> None:
    """Reject a train whose boundary tensors don't describe an open-boundary layout.

    An open-boundary train's two boundary tensors always agree in rank: 2
    vs 2 for an MPS, 3 vs 3 for an MPO. That's checked directly first, since
    it's cheap and unconditional.

    Agreement alone isn't enough: an open-boundary train's boundary rank is
    also always exactly one less than its interior rank, and a periodic train
    would have equal boundary and interior ranks instead. Comparing each end
    against its neighbouring interior tensor catches that, but only when the
    boundary rank itself is plausible (2 to 4); anything further off, like
    rank 0 or rank 8, is left to downstream validation instead of a speculative
    diagnosis.

    A train with fewer than three sites has no interior tensor to compare
    against, so only the first check applies there. That's enough to catch
    a periodic MPO this short (its boundary rank, 4, doesn't match any valid
    two-site layout, so it's still rejected downstream as unrecognised) but
    not a periodic MPS: its boundary tensors have the same rank a valid
    two-site MPO's would (3, since both ends agree with each other), and
    with no interior tensor to tell them apart, it passes through entirely
    unnoticed.

    Args:
        arrays: The site tensors of a single train. An empty sequence is a
            no-op; the downstream layout check handles that case.
        label: Which train this is, for the error message (e.g.
            `"left_tensor"`).

    Raises:
        ValueError: If the two boundary tensors' ranks disagree, or (for
            three or more sites only) if either one is inconsistent with
            its neighbouring interior tensor.
    """
    if not arrays:
        return

    first_ndim = np.ndim(arrays[0])
    last_ndim = np.ndim(arrays[-1])

    if first_ndim != last_ndim:
        msg = (
            f"{label} has inconsistent boundary tensor ranks: rank {first_ndim} "
            f"at the first site vs rank {last_ndim} at the last site."
        )
        raise ValueError(msg)

    if len(arrays) >= _MIN_SITES_FOR_INTERIOR_CHECK:
        _validate_against_interior(first_ndim, np.ndim(arrays[1]), "first", label)
        _validate_against_interior(last_ndim, np.ndim(arrays[-2]), "last", label)


def _validate_against_interior(
    boundary_ndim: int, interior_ndim: int, which: str, label: str
) -> None:
    """Raise unless `boundary_ndim` is exactly one less than `interior_ndim`.

    Only applies when `boundary_ndim` itself falls in the plausible range
    for an open-boundary rank (`_MIN_PLAUSIBLE_BOUNDARY_NDIM` to
    `_PERIODIC_BOUNDARY_MIN_NDIM` inclusive). Outside that range the input
    doesn't resemble a tensor train at all, so this is a no-op and the error
    should be addressed downstream rather than speculating about periodic
    boundary conditions for input that isn't close to any recognised layout.

    Args:
        boundary_ndim: The rank of the boundary site tensor being checked.
        interior_ndim: The rank of its neighbouring interior site tensor.
        which: `"first"` or `"last"`, for the error message.
        label: Which train this is, for the error message (e.g.
            `"left_tensor"`).

    Raises:
        ValueError: If `boundary_ndim` is in the plausible range but is not
            exactly one less than `interior_ndim`.
    """
    if not _MIN_PLAUSIBLE_BOUNDARY_NDIM <= boundary_ndim <= _PERIODIC_BOUNDARY_MIN_NDIM:
        return
    if boundary_ndim != interior_ndim - 1:
        msg = (
            f"{label}'s {which} site tensor has rank {boundary_ndim}, but its "
            f"neighbouring interior site tensor has rank {interior_ndim}. An "
            "open-boundary train's boundary rank must be exactly one less than "
            "its interior rank; only open boundary conditions are supported."
        )
        raise ValueError(msg)
