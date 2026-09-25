"""Compression of a single MPS or MPO.

A thin wrapper around `src_method.stack.src` for one-train stacks:

1. MPO randomized compression.
2. MPS randomized compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .stack import src

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ._plan import Resources


def compress(
    tensor: Sequence[NDArray],
    chi_out: int,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
    seed: int | None = None,
    device: str = "cpu",
    resources: Resources | None = None,
) -> list[NDArray]:
    """Applies the Successive Randomized Compression (SRC) algorithm.

    Equivalent to ``src(tensor, ...)``; see `src_method.stack.src` for the
    conventions. The train type is inferred from the rank of the first site
    tensor:

      1. MPS: `tensor` is an MPS. Results in an MPS.
      2. MPO: `tensor` is an MPO. Results in an MPO.

    Args:
        tensor: The site arrays of the tensor network to compress (MPS or MPO).
        chi_out: The desired maximum bond dimension of the output tensor network.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
            When positive, bonds are trimmed to their effective rank by
            discarding singular values below ``cutoff * sigma_max`` at each
            site during the right-to-left sweep.  The SVD operates on the
            small ``(chi_out, chi_out)`` R factor from QR, so overhead is
            minimal.  Set to 0.0 (default) to keep all bonds at chi_out.
        dtype: The data type for the computation.
        seed: An optional seed for the random number generator.
        device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy).  Requires
            the optional ``cupy`` dependency for GPU execution.
        resources: Memory budgets and scratch space; see `src_method.stack.src`.

    Returns:
        The site arrays of the compressed tensor network (MPS or MPO).

    Raises:
        TypeError: If the input tensor type is unsupported.
        ValueError: If a sub-three-site train is not exactly two sites, or if
            ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    return src(
        tensor,
        chi_out=chi_out,
        cutoff=cutoff,
        dtype=dtype,
        seed=seed,
        device=device,
        resources=resources,
    )
