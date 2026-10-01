"""Site kernels of the SRC sweep and the memory they need.

Every kernel is one of the contractions of the sweep restricted to a batch: a slice
of the sketch index for the environments and the sketch, a slice of the rows of
the new projected environment otherwise. Batching never changes the result beyond
rounding, because the sketch index and those rows are free indices of their
contractions.
"""

from __future__ import annotations

from functools import cache
from itertools import count
from math import prod
from typing import TYPE_CHECKING, NamedTuple

import opt_einsum as oe
from opt_einsum import contract_expression, get_symbol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from opt_einsum.contract import ContractExpression

Shape = tuple[int, ...]


class Equations(NamedTuple):
    """The einsum equations of one sweep, for a fixed stack depth."""

    ltr: str
    rtl_m: str
    rtl_s: str
    first: str


@cache
def equations(depth: int) -> Equations:
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
    return Equations(
        ltr=f"{sketch}{left},{sketch}{up}{down},{layers}->{sketch}{right}",
        rtl_m=f"{sketch}{left},{layers},{eta_right}{right}->{eta_right}{up}{down}{sketch}",
        rtl_s=f"{eta_left}{eta_right}{up}{down},{layers},{eta_right}{right}->{eta_left}{left}",
        first=f"{layers},{eta_right}{right}->{left}{eta_right}{up}{down}",
    )


class _Contractions:
    """Compiled contractions for one sweep, keyed on equation and operand shapes.

    Uniform bulk sites share one entry, so the path is planned once rather than at
    every site. Kept per call: jagged bonds and short last batches add entries that
    are not worth keeping.
    """

    def __init__(self) -> None:
        self._compiled: dict[tuple[str, tuple[Shape, ...]], ContractExpression] = {}

    def __call__(self, eq: str, *operands: NDArray) -> NDArray:
        shapes = tuple(op.shape for op in operands)
        expr = self._compiled.get((eq, shapes))
        if expr is None:
            expr = self._compiled[eq, shapes] = contract_expression(eq, *shapes)
        return expr(*operands)


class SiteKernels:
    """The four contractions of the sweep, each applied to one batch.

    Args:
        depth: The number of layers of the stack.
    """

    def __init__(self, depth: int) -> None:
        self.eqs = equations(depth)
        self._contract = _Contractions()

    def env(self, env: NDArray, omega: NDArray, cores: Sequence[NDArray]) -> NDArray:
        """Advance a batch of sketch columns of the environment by one site.

        Args:
            env: Columns ``lo:hi`` of ``C_j``, shape ``(b, *left_bonds)``.
            omega: The same columns of the site's Gaussian tensor, ``(b, up, down)``.
            cores: The padded cores of the site.

        Returns:
            Columns ``lo:hi`` of ``C_{j+1}``, shape ``(b, *right_bonds)``.
        """
        return self._contract(self.eqs.ltr, env, omega, *cores)

    def sketch(self, env: NDArray, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Sketch a batch of columns of the site's running core.

        Args:
            env: Columns ``lo:hi`` of ``C_j``, shape ``(b, *left_bonds)``.
            cores: The padded cores of the site.
            proj: The projected environment ``S``, ``(eta, *right_bonds)``.

        Returns:
            Columns ``lo:hi`` of the sketch, shape ``(eta, up, down, b)``.
        """
        return self._contract(self.eqs.rtl_m, env, *cores, proj)

    def project(self, eta: NDArray, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Project a batch of rows of the new projected environment.

        Args:
            eta: Rows ``lo:hi`` of the conjugated output core, ``(b, eta, up, down)``.
            cores: The padded cores of the site.
            proj: The projected environment ``S``, ``(eta, *right_bonds)``.

        Returns:
            Rows ``lo:hi`` of the new ``S``, shape ``(b, *left_bonds)``.
        """
        return self._contract(self.eqs.rtl_s, eta, *cores, proj)

    def first(self, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Contract the first site with a batch of rows of ``S``.

        Args:
            cores: The padded cores of the first site.
            proj: Rows ``lo:hi`` of ``S``, ``(b, *right_bonds)``.

        Returns:
            The output core for those rows, ``(*left_bonds, b, up, down)``.
        """
        return self._contract(self.eqs.first, *cores, proj)


@cache
def peak_elements(eq: str, shapes: tuple[Shape, ...]) -> int:
    """Estimate the peak elements a contraction allocates beyond its inputs.

    Walks the path `opt_einsum` picks for these shapes, the one `SiteKernels` runs.
    Each pairwise step holds the intermediates still alive, its output and a
    possible contiguous copy of both operands (``tensordot`` transposes them), so
    the estimate errs on the high side. The final output is included, the inputs
    are not.

    Args:
        eq: The einsum equation.
        shapes: The operand shapes.

    Returns:
        The peak number of elements.
    """
    _, info = oe.contract_path(eq, *shapes, shapes=True)
    sizes = info.size_dict
    # (elements, is_intermediate) for every operand still to be contracted.
    operands = [(prod(shape), False) for shape in shapes]
    peak = 0
    for step in info.contraction_list:
        positions, einsum_str = step[0], step[2]
        live = sum(n for n, is_tmp in operands if is_tmp)
        popped = [operands.pop(i) for i in positions]
        out = prod(sizes[c] for c in einsum_str.split("->")[1])
        peak = max(peak, live + out + sum(n for n, _ in popped))
        operands.append((out, True))
    return peak
