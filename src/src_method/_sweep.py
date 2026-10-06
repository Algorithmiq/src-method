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

from functools import cache
from itertools import count
from math import prod
from time import perf_counter_ns
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import structlog
from opt_einsum import contract_expression, get_symbol

from ._tensor_train import pad, unpad
from .utils import to_numpy, truncated_qr

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import DTypeLike, NDArray
    from opt_einsum.contract import ContractExpression

    from ._tensor_train import TrainKind

logger = structlog.get_logger(__name__)


class _Contractions:
    """Compiled contractions for one sweep, keyed on equation and operand shapes.

    Uniform bulk sites share one entry, so the path is planned once rather than at
    every site. Kept per call: jagged bonds add entries that are not worth keeping.
    """

    def __init__(self) -> None:
        self._compiled: dict[
            tuple[str, tuple[tuple[int, ...], ...]], ContractExpression
        ] = {}

    def __call__(self, eq: str, *operands: NDArray) -> NDArray:
        shapes = tuple(op.shape for op in operands)
        expr = self._compiled.get((eq, shapes))
        if expr is None:
            expr = self._compiled[eq, shapes] = contract_expression(eq, *shapes)
        return expr(*operands)


class _Equations(NamedTuple):
    """The einsum equations of one sweep, for a fixed stack depth."""

    ltr: str
    rtl_m: str
    rtl_s: str
    first: str


@cache
def _equations(depth: int) -> _Equations:
    """Generate the sweep equations for a stack of ``depth`` layers.

    Layer ``i`` at a site carries ``(a_i, b_i, x_i, x_{i+1})``: left and right
    bonds, then its upper and lower physical legs, so that consecutive layers share
    ``x``. The output legs are ``x_0`` (up) and ``x_depth`` (down).
    """
    symbols = map(get_symbol, count())
    sketch, eta_right, eta_left = next(symbols), next(symbols), next(symbols)
    left = "".join(next(symbols) for _ in range(depth))
    right = "".join(next(symbols) for _ in range(depth))
    phys = [next(symbols) for _ in range(depth + 1)]
    up, down = phys[0], phys[-1]
    layers = ",".join(
        f"{left[i]}{right[i]}{phys[i]}{phys[i + 1]}" for i in range(depth)
    )
    return _Equations(
        ltr=f"{sketch}{left},{sketch}{up}{down},{layers}->{sketch}{right}",
        rtl_m=f"{sketch}{left},{layers},{eta_right}{right}->{eta_right}{up}{down}{sketch}",
        rtl_s=f"{eta_left}{eta_right}{up}{down},{layers},{eta_right}{right}->{eta_left}{left}",
        first=f"{layers},{eta_right}{right}->{left}{eta_right}{up}{down}",
    )


def sweep(
    layers: Sequence[Sequence[NDArray]],
    kind: TrainKind,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: DTypeLike = np.float64,
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
    eqs = _equations(depth)
    contract = _Contractions()
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
        C.append(contract(eqs.ltr, C[j], omega, *sites[j]))
    logger.debug("Left-to-right sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

    tms = perf_counter_ns()
    eta_reversed: list[NDArray] = []
    S = xp.ones((1,) * (depth + 1), dtype=dtype)
    for j in range(n_sites - 1, 0, -1):
        # C[-1] is C[j] here; popping it frees each environment once used.
        M = contract(eqs.rtl_m, C.pop(), *sites[j], S)
        rows = M.shape[0] * M.shape[1] * M.shape[2]
        Q = truncated_qr(M.reshape(rows, chi_out), cutoff, xp)
        eta_j = Q.reshape(*M.shape[:3], Q.shape[1]).transpose(3, 0, 1, 2)
        S = contract(eqs.rtl_s, eta_j.conj(), *sites[j], S)
        eta_reversed.append(eta_j)
    first = contract(eqs.first, *sites[0], S)
    eta = [first.reshape(1, *first.shape[depth:]), *reversed(eta_reversed)]
    logger.debug("Right-to-left sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

    return [to_numpy(site) for site in unpad(eta, kind)]
