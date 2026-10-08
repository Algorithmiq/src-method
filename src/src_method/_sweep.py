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
"""

from __future__ import annotations

import logging
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np

from ._kernels import SiteKernels
from ._plan import make_plan, resolve_budgets
from ._sites import SiteSource, padded_shapes, site_bytes
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

    from ._plan import Plan, Resources
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
        resources: Memory budgets and scratch space; detected when ``None``.

    Returns:
        The site arrays of the compressed train in right-canonical form, as numpy
        arrays.
    """
    shapes = padded_shapes(layers)
    # The dtype of the environments and the output: the sketches promoted by the
    # cores, as the contractions would.
    work = np.result_type(dtype, *(site.dtype for layer in layers for site in layer))
    budgets = resolve_budgets(resources, xp)
    plan = make_plan(shapes, site_bytes(layers), chi_out, work, budgets)
    # Guarded: the per-site lists walk the whole plan, wasted work unless logged.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "SRC plan: prefetch=%s, device peak=%d B, host peak=%d B, disk=%d B, "
            "scratch=%s, tiers=%s, batches (env, sketch, project)=%s",
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
    env_shapes = [(chi_out, *(s[0] for s in site)) for site in shapes]
    kernels = SiteKernels(len(layers))
    with (
        device_pool_limit(xp, budgets.device_cap),
        SiteSource(layers, xp, depth=plan.prefetch) as source,
        EnvironmentStore(
            plan, env_shapes, work, xp, budgets.scratch_dir, copy_stream=new_stream(xp)
        ) as store,
    ):
        tms = perf_counter_ns()
        _left_to_right(
            kernels, source, store, plan, chi_out=chi_out, prng=prng, xp=xp, dtype=dtype
        )
        logger.debug("Left-to-right sweep: %.3f s", (perf_counter_ns() - tms) * 1e-9)
        tms = perf_counter_ns()
        eta = _right_to_left(
            kernels,
            source,
            store,
            plan,
            chi_out=chi_out,
            cutoff=cutoff,
            xp=xp,
            dtype=work,
        )
        logger.debug("Right-to-left sweep: %.3f s", (perf_counter_ns() - tms) * 1e-9)
        logger.debug(
            "SRC stalls: sites %.3f s, environments %.3f s",
            source.stall_seconds,
            store.stall_seconds,
        )
        logger.debug("Device pool: %d B", device_pool_bytes(xp))
    return [to_numpy(site) for site in unpad(eta, kind)]


def _left_to_right(
    kernels: SiteKernels,
    source: SiteSource,
    store: EnvironmentStore,
    plan: Plan,
    *,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    dtype: DTypeLike,
) -> None:
    """Build the environments ``C_1 .. C_{n-1}`` into the store."""
    depth = len(kernels.eqs.ltr.split(",")) - 2
    first_env = xp.ones((chi_out,) + (1,) * depth, dtype=dtype)
    n_sites = len(source)
    for j in range(n_sites - 1):
        cores = source[j]
        if plan.prefetch:
            source.prefetch(j + 1)
        up, down = cores[0].shape[2], cores[-1].shape[3]
        omega = gaussian_sketch(prng, (chi_out, up, down), dtype, xp)
        batches = list(_ranges(chi_out, plan.sites[j].env_batch))
        if j > 0:
            store.prefetch(j, batches)
        for lo, hi in batches:
            env = first_env[lo:hi] if j == 0 else store.get(j, lo, hi)
            store.put(j + 1, lo, hi, kernels.env(env, omega[lo:hi], cores))


def _right_to_left(
    kernels: SiteKernels,
    source: SiteSource,
    store: EnvironmentStore,
    plan: Plan,
    *,
    chi_out: int,
    cutoff: float,
    xp: ModuleType,
    dtype: np.dtype,
) -> list[NDArray]:
    """Build the output cores, host-side, from the last site to the first.

    ``dtype`` is the working dtype of the sweep, that of the environments.
    """
    depth = len(kernels.eqs.ltr.split(",")) - 2
    n_sites = len(source)
    eta_reversed: list[NDArray] = []
    proj = xp.ones((1,) * (depth + 1), dtype=dtype)
    for j in range(n_sites - 1, 0, -1):
        cores = source[j]
        if plan.prefetch:
            source.prefetch(j - 1)
        site = plan.sites[j]
        batches = list(_ranges(chi_out, site.sketch_batch))
        store.prefetch(j, batches)
        up, down = cores[0].shape[2], cores[-1].shape[3]
        sketch = xp.empty((proj.shape[0], up, down, chi_out), dtype=dtype)
        for lo, hi in batches:
            sketch[..., lo:hi] = kernels.sketch(store.get(j, lo, hi), cores, proj)
        store.drop(j)
        rows = sketch.shape[0] * up * down
        Q = truncated_qr(sketch.reshape(rows, chi_out), cutoff, xp)
        del sketch
        eta_j = Q.reshape(proj.shape[0], up, down, Q.shape[1]).transpose(3, 0, 1, 2)
        new_proj = xp.empty((Q.shape[1], *(c.shape[0] for c in cores)), dtype=dtype)
        for lo, hi in _ranges(Q.shape[1], site.project_batch):
            new_proj[lo:hi] = kernels.project(eta_j[lo:hi].conj(), cores, proj)
        proj = new_proj
        eta_reversed.append(to_numpy(eta_j))
    cores = source[0]
    up, down = cores[0].shape[2], cores[-1].shape[3]
    first = xp.empty((proj.shape[0], up, down), dtype=dtype)
    for lo, hi in _ranges(proj.shape[0], plan.sites[0].project_batch):
        first[lo:hi] = kernels.first(cores, proj[lo:hi]).reshape(hi - lo, up, down)
    return [to_numpy(first)[None], *reversed(eta_reversed)]
