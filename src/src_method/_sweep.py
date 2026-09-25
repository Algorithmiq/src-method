"""The generic SRC sweep over a stack of tensor trains.

One kernel covers every stack in ket form: ``k`` MPOs, optionally followed by an
MPS. Sites are padded to bulk ``(l, r, u, d)`` views (see
`src_method._tensor_train.pad`), so a single set of einsum equations, generated
from the depth ``k``, serves the boundaries and the bulk alike.

The left-to-right sweep sketches the open physical legs with Gaussian ``omega``
tensors and accumulates the environments ``C``; the sketch index is shared by
every site (a Khatri-Rao sketch). The right-to-left sweep builds the output
through `truncated_qr` while carrying the projected environment ``S``.
"""

from __future__ import annotations

from math import prod
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ._kernels import SiteKernels
from ._tensor_train import pad, unpad
from .utils import to_numpy, truncated_qr

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import NDArray

    from ._tensor_train import TrainKind

logger = structlog.get_logger(__name__)


def sweep(
    layers: Sequence[Sequence[NDArray]],
    kind: TrainKind,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
) -> list[NDArray]:
    """Contract and compress a stack in ket form with one SRC sweep.

    Args:
        layers: MPOs, optionally followed by one MPS, all with the same number
            (at least three) of sites and matching physical legs.
        kind: The kind of the contracted train.
        chi_out: The sketch size, which is the maximum output bond dimension.
        prng: The generator for the Gaussian sketches, always host-side so that a
            seed gives the same draws on every device.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type of the sketches.

    Returns:
        The site arrays of the compressed train in right-canonical form, as numpy
        arrays.
    """
    depth = len(layers)
    n_sites = len(layers[0])
    kernels = SiteKernels(depth)
    # sites[j] holds the padded tensors of every layer at site j.
    sites = list(zip(*(pad([xp.asarray(a) for a in layer]) for layer in layers)))
    logger.debug(
        "Largest environment (elements)",
        size=chi_out
        * max(prod(t.shape[1] for t in sites[j]) for j in range(n_sites - 1)),
    )

    tms = perf_counter_ns()
    # C[j] is the sketched environment of sites 0 .. j-1.
    C = [xp.ones((chi_out,) + (1,) * depth, dtype=dtype)]
    for j in range(n_sites - 1):
        up, down = sites[j][0].shape[2], sites[j][-1].shape[3]
        omega = xp.asarray(prng.normal(size=(chi_out, up, down))).astype(dtype)
        C.append(kernels.env(C[j], omega, sites[j]))
    logger.debug("Left-to-right sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

    tms = perf_counter_ns()
    eta_reversed: list[NDArray] = []
    S = xp.ones((1,) * (depth + 1), dtype=dtype)
    for j in range(n_sites - 1, 0, -1):
        # C[-1] is C[j] here; popping it frees each environment once used.
        M = kernels.sketch(C.pop(), sites[j], S)
        rows = M.shape[0] * M.shape[1] * M.shape[2]
        Q = truncated_qr(M.reshape(rows, chi_out), cutoff, xp)
        eta_j = Q.reshape(*M.shape[:3], Q.shape[1]).transpose(3, 0, 1, 2)
        S = kernels.project(eta_j.conj(), sites[j], S)
        eta_reversed.append(eta_j)
    first = kernels.first(sites[0], S)
    eta = [first.reshape(1, *first.shape[depth:]), *reversed(eta_reversed)]
    logger.debug("Right-to-left sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

    return [to_numpy(site) for site in unpad(eta, kind)]
