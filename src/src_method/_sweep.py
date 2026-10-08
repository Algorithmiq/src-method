"""The generic SRC sweep over a stack of tensor trains.

One kernel covers every stack in ket form: ``k`` MPOs, optionally followed by an
MPS. Sites are padded to bulk ``(l, r, u, d)`` views (see
`src_method._tensor_train.pad`), so a single set of einsum equations, generated
from the depth ``k``, serves the boundaries and the bulk alike.

The left-to-right sweep sketches the open physical legs with Gaussian ``omega``
tensors and accumulates the environments ``C``; the sketch index is shared by
every site (a Khatri-Rao sketch). The right-to-left sweep builds the output
through `truncated_qr` while carrying the projected environment ``S``.

Every contraction runs in batches sized by `src_method._plan.make_plan` to the
memory budgets. The cores are read one site at a time (`SiteSource`) and the
environments live on the device, in host memory or on disk (`EnvironmentStore`).

With several devices (`DeviceGroup`), each owns a contiguous block of the sketch
columns. The left-to-right pass needs no communication: the columns are
independent. At every site of the right-to-left pass, the blocks of the sketch are
gathered on rank 0, which runs the QR and scatters blocks of rows of the output
core; each device projects its rows of the new ``S``, and the rows are gathered on
every device. The random draws, and hence the result, are those of one device.
"""

from __future__ import annotations

import logging
import threading
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np

from ._group import DeviceGroup, blocks
from ._kernels import SiteKernels
from ._plan import make_plan, resolve_budgets, resolve_devices
from ._sites import SiteSource, padded_shapes, site_bytes, sweep_order
from ._store import EnvironmentStore
from ._tensor_train import unpad
from .utils import (
    device_pool_bytes,
    device_pool_limit,
    gaussian_sketch,
    new_stream,
    to_numpy,
    truncated_qr,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from numpy.typing import DTypeLike, NDArray

    from ._plan import Budgets, Plan, Resources
    from ._tensor_train import Site, TrainKind

logger = logging.getLogger(__name__)


def _ranges(n: int, batch: int) -> Iterator[tuple[int, int]]:
    """Split ``range(n)`` into consecutive ``(lo, hi)`` batches."""
    for lo in range(0, n, batch):
        yield lo, min(lo + batch, n)


def sweep(
    layers: Sequence[Sequence[Site]],
    kind: TrainKind,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: DTypeLike,
    resources: Resources | None = None,
) -> list[NDArray]:
    """Contract and compress a stack in ket form with one SRC sweep.

    Args:
        layers: MPOs, optionally followed by one MPS, all with the same number
            (at least three) of sites and matching physical legs. Sites may be any
            array-likes with ``shape``, ``dtype``, ``ndim`` and ``np.asarray``
            support; each is read only when the sweep reaches it.
        kind: The kind of the contracted train.
        chi_out: The sketch size, which is the maximum output bond dimension.
        prng: The generator for the Gaussian sketches, always host-side so that a
            seed gives the same draws on every device.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type of the sketches.
        resources: Memory budgets, scratch space and devices; detected when
            ``None``.

    Returns:
        The site arrays of the compressed train in right-canonical form, as numpy
        arrays.
    """
    shapes = padded_shapes(layers)
    # The dtype of the environments and the output: the sketches promoted by the
    # cores, as the contractions would.
    work = np.result_type(dtype, *(site.dtype for layer in layers for site in layer))
    devices = resolve_devices(resources, xp)
    group = DeviceGroup(xp, devices)
    budgets = resolve_budgets(resources, xp, devices)
    plan = make_plan(
        shapes, site_bytes(layers), chi_out, work, budgets, devices=group.size
    )
    # Guarded: the per-site lists walk the whole plan, wasted work unless logged.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "SRC plan: devices=%s, prefetch=%s, device peak=%d B, host peak=%d B, "
            "disk=%d B, scratch=%s, tiers=%s, batches (env, sketch, project)=%s",
            list(devices),
            plan.prefetch,
            plan.device_peak,
            plan.host_peak,
            plan.disk_bytes,
            budgets.scratch_dir if plan.disk_bytes else None,
            [site.tier for site in plan.sites],
            [
                (site.env_batch, site.sketch_batch, site.project_batch)
                for site in plan.sites
            ],
        )
    sweep_ = _Sweep(
        group,
        plan,
        budgets,
        shapes,
        chi_out=chi_out,
        cutoff=cutoff,
        sketches=_Sketches(prng, chi_out, dtype, group.size),
        work=work,
        depth=len(layers),
    )
    with SiteSource(
        layers, xp, sweep_order(len(shapes)), depth=plan.prefetch, group=group
    ) as source:
        eta = group.run(lambda rank: sweep_.run(rank, source))[0]
    assert eta is not None  # noqa: S101  (rank 0 returns the cores)
    return [to_numpy(site) for site in unpad(eta, kind)]


class _Sketches:
    """The Gaussian tensors of the left-to-right pass, shared by the devices.

    Every device asks for the sites in order, and the first one at a site draws its
    tensor, so the draws come in the order of a single device, whatever the
    timing. A tensor is dropped once every device took it.
    """

    def __init__(
        self, prng: np.random.Generator, chi: int, dtype: DTypeLike, ranks: int
    ) -> None:
        self._prng, self._chi, self._dtype, self._ranks = prng, chi, dtype, ranks
        self._drawn: dict[int, tuple[np.ndarray, list[int]]] = {}
        self._lock = threading.Lock()

    def get(self, j: int, up: int, down: int) -> np.ndarray:
        """Return the host tensor ``(chi, up, down)`` of site ``j``."""
        with self._lock:
            if j not in self._drawn:
                omega = gaussian_sketch(
                    self._prng, (self._chi, up, down), self._dtype, np
                )
                self._drawn[j] = (omega, [self._ranks])
            omega, left = self._drawn[j]
            left[0] -= 1
            if left[0] == 0:
                del self._drawn[j]
        return omega


class _Sweep:
    """The per-device part of a sweep; `run` executes it on one rank."""

    def __init__(
        self,
        group: DeviceGroup,
        plan: Plan,
        budgets: Budgets,
        shapes: Sequence[tuple[tuple[int, ...], ...]],
        *,
        chi_out: int,
        cutoff: float,
        sketches: _Sketches,
        work: np.dtype,
        depth: int,
    ) -> None:
        self.group, self.plan, self.budgets, self.shapes = group, plan, budgets, shapes
        self.chi_out, self.cutoff, self.sketches = chi_out, cutoff, sketches
        self.work, self.depth = work, depth
        self.columns = blocks(chi_out, group.size)

    def _tag(self, rank: int) -> str:
        if self.group.size == 1:
            return ""
        return f"Device {self.group.devices[rank]}: "

    def run(self, rank: int, source: SiteSource) -> list[NDArray] | None:
        """Run both passes on ``rank``; rank 0 returns the output cores."""
        xp, tag = self.group.xp, self._tag(rank)
        lo, hi = self.columns[rank]
        env_shapes = [(hi - lo, *(s[0] for s in site)) for site in self.shapes]
        kernels = SiteKernels(self.depth)
        with (
            device_pool_limit(xp, self.budgets.device_cap),
            EnvironmentStore(
                self.plan,
                env_shapes,
                self.work,
                xp,
                self.budgets.scratch_dir,
                copy_stream=new_stream(xp),
            ) as store,
        ):
            tms = perf_counter_ns()
            self._left_to_right(rank, kernels, source, store)
            logger.debug(
                "%sLeft-to-right sweep: %.3f s", tag, (perf_counter_ns() - tms) * 1e-9
            )
            tms = perf_counter_ns()
            eta = self._right_to_left(rank, kernels, source, store)
            logger.debug(
                "%sRight-to-left sweep: %.3f s", tag, (perf_counter_ns() - tms) * 1e-9
            )
            logger.debug(
                "%sSRC stalls: sites %.3f s, environments %.3f s",
                tag,
                source.stall_seconds[rank],
                store.stall_seconds,
            )
            logger.debug("%sDevice pool: %d B", tag, device_pool_bytes(xp))
        return eta

    def _left_to_right(
        self,
        rank: int,
        kernels: SiteKernels,
        source: SiteSource,
        store: EnvironmentStore,
    ) -> None:
        """Build this device's columns of the environments ``C_1 .. C_{n-1}``."""
        xp, plan = self.group.xp, self.plan
        col_lo, col_hi = self.columns[rank]
        n_local = col_hi - col_lo
        first_env = xp.ones((n_local,) + (1,) * self.depth, dtype=self.work)
        for j in range(len(source) - 1):
            cores = source.next(rank)
            up, down = cores[0].shape[2], cores[-1].shape[3]
            omega = xp.asarray(self.sketches.get(j, up, down)[col_lo:col_hi])
            batches = list(_ranges(n_local, plan.sites[j].env_batch))
            if j > 0:
                store.prefetch(j, batches)
            for lo, hi in batches:
                env = first_env[lo:hi] if j == 0 else store.get(j, lo, hi)
                store.put(j + 1, lo, hi, kernels.env(env, omega[lo:hi], cores))

    def _right_to_left(
        self,
        rank: int,
        kernels: SiteKernels,
        source: SiteSource,
        store: EnvironmentStore,
    ) -> list[NDArray] | None:
        """Build the output cores, host-side on rank 0, from the last site down."""
        xp, plan, group, dtype = self.group.xp, self.plan, self.group, self.work
        root = rank == 0
        col_lo, col_hi = self.columns[rank]
        n_local = col_hi - col_lo
        eta_reversed: list[NDArray] = []
        proj = xp.ones((1,) * (self.depth + 1), dtype=dtype)
        for j in range(len(source) - 1, 0, -1):
            cores = source.next(rank)
            site = plan.sites[j]
            batches = list(_ranges(n_local, site.sketch_batch))
            store.prefetch(j, batches)
            up, down = cores[0].shape[2], cores[-1].shape[3]
            # Sketch columns first, so that each device's block is contiguous.
            local = xp.empty((n_local, proj.shape[0], up, down), dtype=dtype)
            for lo, hi in batches:
                part = kernels.sketch(store.get(j, lo, hi), cores, proj)
                local[lo:hi] = xp.moveaxis(part, -1, 0)
            store.drop(j)
            sketch = group.gather(rank, local)
            del local
            eta_full = None
            if root:
                assert sketch is not None  # noqa: S101  (gathered on rank 0)
                rows = proj.shape[0] * up * down
                Q = truncated_qr(sketch.reshape(self.chi_out, rows).T, self.cutoff, xp)
                del sketch
                eta_full = xp.ascontiguousarray(Q.T).reshape(
                    -1, proj.shape[0], up, down
                )
                del Q
            eta_local = group.scatter(rank, eta_full)
            if eta_full is not None:
                eta_reversed.append(to_numpy(eta_full))
                del eta_full
            new_local = xp.empty(
                (eta_local.shape[0], *(c.shape[0] for c in cores)), dtype=dtype
            )
            for lo, hi in _ranges(eta_local.shape[0], site.project_batch):
                new_local[lo:hi] = kernels.project(eta_local[lo:hi].conj(), cores, proj)
            del proj, eta_local
            proj = group.allgather(rank, new_local)
            del new_local
        # Every device takes the first site, so that the source can release it.
        cores = source.next(rank)
        if not root:
            return None
        up, down = cores[0].shape[2], cores[-1].shape[3]
        first = xp.empty((proj.shape[0], up, down), dtype=dtype)
        for lo, hi in _ranges(proj.shape[0], plan.sites[0].project_batch):
            first[lo:hi] = kernels.first(cores, proj[lo:hi]).reshape(hi - lo, up, down)
        return [to_numpy(first)[None], *reversed(eta_reversed)]
