"""Argument and layout checks shared by the public entry points.

A malformed call should fail at once with a message naming the culprit, rather than
as an opaque shape error deep inside the sweep.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._tensor_train import Site

__all__ = ["validate_chi_out", "validate_cutoff", "validate_open_boundary"]


def validate_chi_out(chi_out: object) -> int:
    """Reject a non-integer or non-positive output bond dimension.

    Any `numbers.Integral` is accepted, so NumPy integers pass; `bool` does not.

    Args:
        chi_out: The requested output bond dimension.

    Returns:
        ``chi_out`` as a plain `int`.

    Raises:
        TypeError: If `chi_out` is not an integer.
        ValueError: If `chi_out` is not positive.
    """
    if isinstance(chi_out, bool) or not isinstance(chi_out, numbers.Integral):
        msg = f"chi_out must be an integer, got {chi_out!r}."
        raise TypeError(msg)
    if chi_out < 1:
        msg = f"chi_out must be positive, got {chi_out!r}."
        raise ValueError(msg)
    return int(chi_out)


def validate_cutoff(cutoff: float) -> None:
    """Reject a cutoff outside ``[0.0, 1.0)``.

    A cutoff of 1 or more would discard every singular value, and a negative one
    would silently mean "no truncation".

    Args:
        cutoff: The relative singular-value cutoff.

    Raises:
        ValueError: If `cutoff` is not a real number in ``[0.0, 1.0)``.
    """
    if not isinstance(cutoff, numbers.Real) or not 0.0 <= cutoff < 1.0:
        msg = f"cutoff must be in [0.0, 1.0), got {cutoff!r}."
        raise ValueError(msg)


def validate_open_boundary(
    train: Sequence[Site], boundary_ndim: int, index: int
) -> None:
    """Check that a train has the layout of an open-boundary MPS or MPO.

    An open-boundary train has rank ``boundary_ndim`` at its two ends and one more
    (the second bond) at every interior site. A periodic train has the interior rank
    everywhere; read from its first site it is misclassified as a train of the
    next-lower rank, which this check unmasks.

    A two-site train has no interior site, so a periodic one cannot be told apart
    from an open one of the same rank and is not caught.

    Args:
        train: The site arrays of the train.
        boundary_ndim: The rank of the boundary sites, as inferred from the first
            site.
        index: The position of the train in the stack, for the error message.

    Raises:
        ValueError: If the ranks do not follow the open-boundary pattern.
    """
    expected = [boundary_ndim, *[boundary_ndim + 1] * (len(train) - 2), boundary_ndim]
    ranks = [np.ndim(site) for site in train]
    if len(train) > 1 and ranks != expected:
        msg = (
            f"Train {index} has site ranks {ranks}, expected {expected}: only "
            "open boundary conditions are supported."
        )
        raise ValueError(msg)
