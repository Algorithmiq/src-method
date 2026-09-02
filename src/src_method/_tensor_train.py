"""Array-list tensor-train conventions and exact small-system primitives.

Tensor trains are plain lists of arrays, one per site.  The index ordering
matches the default `quimb` layout, so a result can be handed straight to
``qtn.MatrixProductState(arrays)`` / ``qtn.MatrixProductOperator(arrays)``
without any permutation:

* MPS: ``(bond_r, phys)``, ``(bond_l, bond_r, phys)``, ..., ``(bond_l, phys)``
* MPO: ``(bond_r, up, down)``, ``(bond_l, bond_r, up, down)``, ...,
  ``(bond_l, up, down)``

The SRC sweep needs at least three sites, so two-site trains are handled here
instead.  At that size the whole network fits in a single dense matrix, and one
exact SVD is both cheaper and more accurate than a randomized sketch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from opt_einsum import contract

from .utils import to_numpy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Minimum number of sites for which the randomized SRC sweep is defined.
MIN_SRC_SITES = 3

# The only sub-``MIN_SRC_SITES`` size the exact path can handle.
_EXACT_SITES = 2

# Rank of a boundary (first / last) site tensor, which identifies the train type.
_MPS_BOUNDARY_NDIM = 2
_MPO_BOUNDARY_NDIM = 3

TrainKind = Literal["mps", "mpo"]

__all__ = [
    "MIN_SRC_SITES",
    "TrainKind",
    "check_exact_supported",
    "exact_apply",
    "exact_compress",
    "infer_kind",
]


def infer_kind(arrays: Sequence[NDArray]) -> TrainKind | None:
    """Classify a tensor train from the rank of its first site tensor.

    A boundary site carries one bond index plus either a single physical
    index (MPS) or an upper/lower pair (MPO), so the rank is unambiguous.

    Args:
        arrays: The site tensors of the train.

    Returns:
        ``"mps"``, ``"mpo"``, or ``None`` if the layout is unrecognised.
    """
    if len(arrays) == 0:
        return None
    ndim = np.ndim(arrays[0])
    if ndim == _MPS_BOUNDARY_NDIM:
        return "mps"
    if ndim == _MPO_BOUNDARY_NDIM:
        return "mpo"
    return None


def check_exact_supported(n_sites: int) -> None:
    """Reject sub-``MIN_SRC_SITES`` trains the exact path cannot handle.

    Called at the public boundary before the fallback is announced, so that a
    degenerate train raises instead of first logging a misleading warning.

    Args:
        n_sites: The number of sites in the train.

    Raises:
        ValueError: If the train does not have exactly two sites.
    """
    if n_sites != _EXACT_SITES:
        msg = (
            f"Expected a two-site tensor train, got {n_sites} site(s). "
            "Single-site trains are degenerate; use three or more sites for SRC."
        )
        raise ValueError(msg)


def exact_compress(
    arrays: Sequence[NDArray], chi_out: int, kind: TrainKind
) -> list[NDArray]:
    """Compress a two-site train exactly via a single truncated SVD.

    Site counts are validated by the caller via `check_exact_supported`.

    Args:
        arrays: The two site tensors of the train.
        chi_out: The maximum bond dimension to keep.
        kind: Whether the train is an ``"mps"`` or an ``"mpo"``.

    Returns:
        The compressed train, in right-canonical form, as numpy arrays.
    """
    # The dense SVD is host-side, so accept device arrays like the sweep does.
    arrays = [to_numpy(arr) for arr in arrays]
    if kind == "mps":
        # (b, p0) x (b, p1) -> (p0, p1)
        theta = contract("ab,ac->bc", arrays[0], arrays[1])
        left, right = _truncated_svd(theta, chi_out)
        return [left.T, right]

    # (b, u0, d0) x (b, u1, d1) -> (u0, d0, u1, d1)
    theta = contract("aij,akl->ijkl", arrays[0], arrays[1])
    up_l, down_l, up_r, down_r = theta.shape
    left, right = _truncated_svd(theta.reshape(up_l * down_l, up_r * down_r), chi_out)
    rank = left.shape[1]
    return [
        left.reshape(up_l, down_l, rank).transpose(2, 0, 1),
        right.reshape(rank, up_r, down_r),
    ]


def exact_apply(
    left_tensor: Sequence[NDArray],
    right_tensor: Sequence[NDArray],
    chi_out: int,
    kind: TrainKind,
) -> list[NDArray]:
    """Contract and compress two two-site trains exactly.

    The MPO on the left is contracted site-wise with the right train, fusing
    the two bond indices, and the result is compressed with a single SVD.
    Site counts are validated by the caller via `check_exact_supported`.

    Args:
        left_tensor: The two site tensors of the left MPO.
        right_tensor: The two site tensors of the right MPS or MPO.
        chi_out: The maximum bond dimension to keep.
        kind: Whether ``right_tensor`` is an ``"mps"`` or an ``"mpo"``.

    Returns:
        The compressed product, in right-canonical form, as numpy arrays.
    """
    left_tensor = [to_numpy(arr) for arr in left_tensor]
    right_tensor = [to_numpy(arr) for arr in right_tensor]
    if kind == "mps":
        # Contract the MPO lower leg with the MPS physical leg, fusing both bonds.
        product = [
            contract("aij,bj->abi", left_tensor[i], right_tensor[i]).reshape(
                -1, left_tensor[i].shape[1]
            )
            for i in range(2)
        ]
    else:
        product = [
            contract("aij,bjk->abik", left_tensor[i], right_tensor[i]).reshape(
                -1, left_tensor[i].shape[1], right_tensor[i].shape[2]
            )
            for i in range(2)
        ]
    return exact_compress(product, chi_out, kind)


def _truncated_svd(theta: NDArray, chi_out: int) -> tuple[NDArray, NDArray]:
    """Split a matrix as ``(U @ diag(S), Vh)``, keeping at most ``chi_out`` values."""
    U, S, Vh = np.linalg.svd(theta, full_matrices=False)
    rank = min(chi_out, S.size)
    return U[:, :rank] * S[:rank], Vh[:rank]
