"""Successive Randomized Compression of a stack of tensor trains.

A stack ``T_1 . T_2 . ... . T_m`` is an ordered sequence of trains contracted
along their physical legs and compressed in a single SRC sweep. `apply` and
`compress` are the two-train and one-train special cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from ._sweep import sweep
from ._tensor_train import (
    MIN_SRC_SITES,
    check_exact_supported,
    exact_stack,
    normalize_stack,
)
from .utils import default_rng, get_xp, setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

setup_logging()
logger = structlog.get_logger(__name__)

LOG_WARN_SMALL = (
    "The current SRC implementation targets tensor networks with 3 or more sites. "
    "Defaulting to an exact SVD-based contraction-compression."
)


def src(
    *trains: Sequence[NDArray],
    chi_out: int,
    cutoff: float = 0.0,
    dtype: type = np.float64,
    seed: int | None = None,
    device: str = "cpu",
) -> list[NDArray]:
    """Contract a stack of tensor trains and compress the result with SRC.

    The stack ``src(T_1, T_2, ..., T_m)`` is the product ``T_1 T_2 ... T_m`` in
    mathematical order: ``T_m`` acts first. Trains are plain lists of per-site
    arrays in the default `quimb` layout, their kind inferred from the rank of the
    first site tensor. Each contraction joins the ``d`` leg of a train with the
    ``u`` (or physical) leg of the next. Accepted stacks:

    - one MPS or one MPO: plain compression;
    - ``A_1, ..., A_k``: an MPO product, giving an MPO;
    - ``A_1, ..., A_k, psi``: an MPO product applied to a ket, giving an MPS;
    - ``phi, A_1, ..., A_k``: a bra times an MPO product, giving an MPS on the
      ``d`` leg of ``A_k``.

    A leading MPS is contracted without conjugation, as the row vector
    ``phi^T A_1 ... A_k``. For the physical bra ``<psi| A_1 ... A_k``, pass
    ``[t.conj() for t in psi]``; the result then pairs with a ket by plain
    contraction.

    The per-site cost grows with ``chi_out**2`` times the product of the layer bond
    dimensions, so one sweep pays off for shallow stacks of thin layers (for example
    two or three Trotter layers). Apply anything else pairwise.

    Args:
        *trains: The site arrays of each train, in mathematical order.
        chi_out: The desired maximum bond dimension of the output train.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
            When positive, bonds are trimmed to their effective rank by
            discarding singular values below ``cutoff * sigma_max`` at each
            site during the right-to-left sweep. Set to 0.0 (default) to keep
            all bonds at ``chi_out``. Ignored for two-site stacks, which are
            contracted and truncated to ``chi_out`` exactly.
        dtype: The data type for the computation.
        seed: An optional seed for the random number generator.
        device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy). Requires
            the optional ``cupy`` dependency for GPU execution.

    Returns:
        The site arrays of the compressed train (MPS or MPO), in right-canonical
        form, as numpy arrays (host-side, whatever the ``device``).

    Raises:
        TypeError: If a train has an unrecognised layout or an MPS sits anywhere
            other than at one end of the stack.
        ValueError: If the stack is empty, if the trains differ in length or in
            the physical dimensions they join, if a sub-three-site stack is not
            exactly two sites, or if ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    xp = get_xp(device)
    prng = default_rng(seed)
    layers, kind = normalize_stack(trains)

    n_sites = len(layers[0])
    if n_sites < MIN_SRC_SITES:
        check_exact_supported(n_sites)
        logger.warning(LOG_WARN_SMALL)
        return exact_stack(layers, chi_out, kind)

    logger.info(
        "Starting SRC",
        n_sites=n_sites,
        depth=len(layers),
        output=kind,
        device=xp.__name__,
    )
    result = sweep(layers, kind, chi_out, prng, xp, cutoff=cutoff, dtype=dtype)
    logger.info("SRC complete.")
    return result
