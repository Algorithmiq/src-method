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
    "exact_compress",
    "exact_stack",
    "infer_kind",
    "normalize_stack",
    "pad",
    "transpose_mpo",
    "unpad",
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


def _truncated_svd(theta: NDArray, chi_out: int) -> tuple[NDArray, NDArray]:
    """Split a matrix as ``(U @ diag(S), Vh)``, keeping at most ``chi_out`` values."""
    U, S, Vh = np.linalg.svd(theta, full_matrices=False)
    rank = min(chi_out, S.size)
    return U[:, :rank] * S[:rank], Vh[:rank]


def pad(train: Sequence[NDArray]) -> list[NDArray]:
    """View every site of a train as a bulk MPO tensor ``(l, r, u, d)``.

    Boundary sites gain a size-1 outer bond and MPS sites a size-1 ``d`` leg, so a
    single contraction pattern covers every site of every train kind. Only views
    are created: the input arrays are neither copied nor mutated.

    Args:
        train: The site tensors of an MPS or MPO with at least two sites.

    Returns:
        The rank-4 views, one per site.
    """
    kind = infer_kind(train)
    last = len(train) - 1
    padded = []
    for i, site in enumerate(train):
        view = site[..., None] if kind == "mps" else site
        if i == 0:
            view = view[None]
        if i == last:
            view = view[:, None]
        padded.append(view)
    return padded


def unpad(train: Sequence[NDArray], kind: TrainKind) -> list[NDArray]:
    """Invert `pad`: drop the size-1 outer bonds and, for an MPS, the ``d`` leg.

    Args:
        train: Rank-4 ``(l, r, u, d)`` site tensors with at least two sites.
        kind: The layout to restore.

    Returns:
        The site tensors in the unpadded `quimb` layout, as views.
    """
    last = len(train) - 1
    unpadded = []
    for i, site in enumerate(train):
        view = site[..., 0] if kind == "mps" else site
        if i == last:
            view = view[:, 0]
        if i == 0:
            view = view[0]
        unpadded.append(view)
    return unpadded


def transpose_mpo(train: Sequence[NDArray]) -> list[NDArray]:
    """Transpose an MPO by swapping its ``u`` and ``d`` legs on every site (views)."""
    return [site.swapaxes(-2, -1) for site in train]


def normalize_stack(
    trains: Sequence[Sequence[NDArray]],
) -> tuple[list[Sequence[NDArray]], TrainKind]:
    """Validate a stack and rewrite it in ket form.

    A stack ``T_1 . T_2 . ... . T_m`` is contracted along the physical legs, the
    ``d`` leg of each train joining the ``u`` (or MPS) leg of the next. Every train
    is an MPO, except that an MPS may come first (a bra) or last (a ket), never
    both. A leading MPS is a row vector used without conjugation, so the bra stack
    ``[phi, A_1, ..., A_k]`` equals the ket stack ``[A_k^T, ..., A_1^T, phi]``,
    which is what this returns.

    Args:
        trains: The trains of the stack, in mathematical order.

    Returns:
        The stack in ket form (MPOs, optionally followed by one MPS) and the kind of
        the contracted train.

    Raises:
        ValueError: If the stack is empty, if the trains differ in length, or if
            adjacent trains have mismatched physical dimensions.
        TypeError: If a train has an unrecognised layout or an MPS sits anywhere
            other than at one end of the stack.
    """
    if len(trains) == 0:
        msg = "Expected at least one tensor train."
        raise ValueError(msg)
    kinds = [infer_kind(train) for train in trains]
    _check_roles(kinds)
    sizes = [len(train) for train in trains]
    if len(set(sizes)) > 1:
        msg = f"All tensor trains must have the same number of sites, got {sizes}."
        raise ValueError(msg)
    _check_physical_dims(trains, kinds)

    if len(trains) > 1 and kinds[0] == "mps":
        bra, *mpos = trains
        return [*(transpose_mpo(mpo) for mpo in reversed(mpos)), bra], "mps"
    return list(trains), "mps" if "mps" in kinds else "mpo"


def _check_roles(kinds: Sequence[TrainKind | None]) -> None:
    """Reject unrecognised layouts and misplaced MPSs."""
    if None in kinds:
        msg = (
            f"Unsupported tensor network layout for train {kinds.index(None)}: "
            "expected an MPS or MPO given as a list of per-site arrays."
        )
        raise TypeError(msg)
    if len(kinds) > 1 and ("mps" in kinds[1:-1] or kinds[0] == kinds[-1] == "mps"):
        msg = (
            f"Unsupported stack {kinds}: every train must be an MPO, except that an "
            "MPS may come first (bra) or last (ket), not both."
        )
        raise TypeError(msg)


def _check_physical_dims(
    trains: Sequence[Sequence[NDArray]], kinds: Sequence[TrainKind | None]
) -> None:
    """Check that the legs joined between adjacent trains agree at every site."""
    for i in range(len(trains) - 1):
        # The outgoing leg is last for an MPO (d) and for a leading MPS alike.
        incoming = -1 if kinds[i + 1] == "mps" else -2
        for site, (upper, lower) in enumerate(zip(trains[i], trains[i + 1])):
            if upper.shape[-1] != lower.shape[incoming]:
                msg = (
                    f"Physical dimension mismatch at site {site} between trains "
                    f"{i} and {i + 1}: {upper.shape[-1]} != {lower.shape[incoming]}."
                )
                raise ValueError(msg)


def exact_stack(
    layers: Sequence[Sequence[NDArray]], chi_out: int, kind: TrainKind
) -> list[NDArray]:
    """Contract and compress a two-site stack exactly.

    At each site the layers are folded into one, fusing their bonds, and the
    product is compressed with a single SVD. Site counts are validated by the
    caller via `check_exact_supported`.

    Args:
        layers: A stack in ket form, as returned by `normalize_stack`.
        chi_out: The maximum bond dimension to keep.
        kind: The kind of the contracted train.

    Returns:
        The compressed product, in right-canonical form, as numpy arrays.
    """
    padded = [pad([to_numpy(site) for site in layer]) for layer in layers]
    product = padded[-1]
    for layer in reversed(padded[:-1]):
        product = [_fuse(upper, lower) for upper, lower in zip(layer, product)]
    return exact_compress(unpad(product, kind), chi_out, kind)


def _fuse(upper: NDArray, lower: NDArray) -> NDArray:
    """Contract ``(l1, r1, u, x) . (l2, r2, x, d)`` into ``(l1 l2, r1 r2, u, d)``."""
    l1, r1, up, _ = upper.shape
    l2, r2, _, down = lower.shape
    return contract("abux,cdxv->acbduv", upper, lower).reshape(
        l1 * l2, r1 * r2, up, down
    )
