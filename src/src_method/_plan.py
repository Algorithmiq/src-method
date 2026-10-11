"""Memory planning for the SRC sweep: budgets, batch sizes and environment tiers.

`make_plan` is a pure function of the core shapes, the sketch size, the dtype and
the budgets, so the plan of a run can be inspected, and tested, without loading any
data or touching a GPU.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._kernels import equations, peak_elements
from .utils import device_memory, host_memory_available, is_host

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence
    from types import ModuleType

Tier = Literal["device", "host", "disk"]
Shape = tuple[int, ...]

# Device memory outside the plan: cuBLAS/cuSOLVER workspaces and pool fragmentation.
# Detection keeps it free, and CuPy's pool may grow into it during the sweep.
GPU_MARGIN_FRACTION = 0.10
GPU_MARGIN_MIN = 2**30
HOST_MARGIN_FRACTION = 0.10
DISK_MARGIN_FRACTION = 0.05
# Batch size used to decide how much of the device can hold environments.
PREFERRED_BATCH = 512
# Batches are rounded down to a multiple of this for efficient GEMMs.
GEMM_MULTIPLE = 32

_UNITS = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?\s*$", re.IGNORECASE)


def parse_size(value: int | str) -> int:
    """Convert a byte count or a size string to bytes.

    Args:
        value: A non-negative integer, or a string such as ``"36GB"`` (decimal,
            ``36 * 10**9``) or ``"36GiB"`` (binary, ``36 * 2**30``).

    Returns:
        The size in bytes.

    Raises:
        TypeError: If the value is neither an integer nor a string.
        ValueError: If the value is negative or not a recognised size.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"Expected a byte count or a size string such as '36GB', got {value!r}."
        raise TypeError(msg)
    if isinstance(value, int):
        if value < 0:
            msg = f"Expected a non-negative byte count, got {value}."
            raise ValueError(msg)
        return value
    match = _SIZE.match(value)
    if match is None:
        msg = f"Expected a size string such as '36GB' or '36GiB', got {value!r}."
        raise ValueError(msg)
    number, unit = match.groups()
    return int(float(number) * _UNITS[(unit or "B").upper()])


@dataclass(frozen=True)
class Resources:
    """Memory budgets and scratch space for one `src` call.

    Every field left as ``None`` is detected when the call starts.

    Attributes:
        gpu_memory: Device memory the sweep plans to use, as a byte count or a
            size string (``"36GB"``, ``"36GiB"``). Defaults to the free device
            memory minus a margin of ``max(10%, 1 GiB)`` of the device. CuPy's pool
            is capped at the budget plus that margin, which absorbs fragmentation.
            Ignored on the CPU.
        host_memory: Host memory the sweep may use. Defaults to ``MemAvailable``
            minus 10%. On the CPU it covers the working set as well.
        scratch_dir: Directory for environments that fit in neither budget,
            ideally on node-local disk. Defaults to ``tempfile.gettempdir()``,
            which honours ``$TMPDIR``.
    """

    gpu_memory: int | str | None = None
    host_memory: int | str | None = None
    scratch_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        """Validate the explicit budgets.

        Raises:
            TypeError: If a budget is neither an integer nor a string.
            ValueError: If a budget is not a recognised size.
        """
        for value in (self.gpu_memory, self.host_memory):
            if value is not None:
                parse_size(value)


@dataclass(frozen=True)
class Budgets:
    """The resolved budgets of one sweep, in bytes.

    Attributes:
        device: Bytes for the working set and the device tier.
        host: Bytes for the output, the staging buffers and the host tier.
        disk: Bytes free in ``scratch_dir``.
        scratch_dir: Where the disk tier lives.
        unified: Whether device and host memory are the same (the CPU backend).
        device_cap: The cap on CuPy's pool during the sweep: ``device`` plus the
            margin for workspaces and fragmentation, which the plan leaves out.
            ``None`` leaves the pool uncapped, as on the CPU backend.
    """

    device: int
    host: int
    disk: int
    scratch_dir: Path
    unified: bool
    device_cap: int | None = None


def resolve_budgets(resources: Resources | None, xp: ModuleType) -> Budgets:
    """Turn `Resources` into byte budgets, detecting those left unset.

    Args:
        resources: The requested budgets, or ``None`` to detect all of them.
        xp: Array module (``numpy`` or ``cupy``).

    Returns:
        The budgets.
    """
    resources = resources or Resources()
    scratch = (
        Path(resources.scratch_dir)
        if resources.scratch_dir is not None
        else Path(tempfile.gettempdir())
    )
    if resources.host_memory is not None:
        host = parse_size(resources.host_memory)
    else:
        host = int(host_memory_available() * (1 - HOST_MARGIN_FRACTION))
    unified = is_host(xp)
    cap = None
    if unified:
        device = host
    else:
        available, total = device_memory(xp)
        margin = max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
        if resources.gpu_memory is not None:
            device = max(parse_size(resources.gpu_memory), 0)
        else:
            device = max(available - margin, 0)
        # The plan counts the bytes in use, but the pool limit applies to every
        # block the pool holds, including split blocks that are partly free.
        cap = device + margin
    free = shutil.disk_usage(_existing_parent(scratch)).free
    disk = int(free * (1 - DISK_MARGIN_FRACTION))
    return Budgets(device, max(host, 0), disk, scratch, unified=unified, device_cap=cap)


def _existing_parent(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path


@dataclass(frozen=True)
class SitePlan:
    """How one site is processed.

    Attributes:
        env_batch: Sketch columns per environment step (0 at the last site).
        sketch_batch: Sketch columns per sketch step (0 at the first site).
        project_batch: Rows per projection step, or per first-site step at site 0.
        tier: Where ``C_j`` is kept; site 0 stores no environment.
    """

    env_batch: int
    sketch_batch: int
    project_batch: int
    tier: Tier


@dataclass(frozen=True)
class Plan:
    """The memory plan of one sweep.

    Attributes:
        sites: One entry per site.
        prefetch: How many sites ahead the cores are loaded (0 or 1).
        device_peak: Estimated peak device bytes.
        host_peak: Estimated peak host bytes, beyond the inputs.
        disk_bytes: Bytes spilled to the scratch directory.
    """

    sites: tuple[SitePlan, ...]
    prefetch: int
    device_peak: int
    host_peak: int
    disk_bytes: int


class _Site:
    """The memory model of one site, in bytes."""

    def __init__(
        self,
        j: int,
        shapes: tuple[Shape, ...],
        core_bytes: int,
        *,
        chi: int,
        itemsize: int,
        n_sites: int,
    ) -> None:
        self.j, self.shapes, self.core_bytes, self.chi, self.e = (
            j,
            shapes,
            core_bytes,
            chi,
            itemsize,
        )
        self.eqs = equations(len(shapes))
        self.left = tuple(s[0] for s in shapes)
        self.right = tuple(s[1] for s in shapes)
        self.a, self.b = prod(self.left), prod(self.right)
        self.up, self.down = shapes[0][2], shapes[-1][3]
        self.p = self.up * self.down
        # Rows of the projected environment S that enters site j right-to-left.
        self.eta = 1 if j == n_sites - 1 else chi
        self.env_bytes = chi * self.a * itemsize

    def env(self, b: int, prefetch: int, *, staged_in: bool, staged_out: bool) -> int:
        """Bytes of one environment step on ``b`` columns."""
        e, chi = self.e, self.chi
        fixed = self.core_bytes * (1 + prefetch) + chi * self.p * e
        slices = (1 + staged_in) * b * self.a * e + staged_out * b * self.b * e
        peak = peak_elements(
            self.eqs.ltr, ((b, *self.left), (b, self.up, self.down), *self.shapes)
        )
        return fixed + slices + peak * e

    def sketch(self, b: int, prefetch: int, *, staged: bool) -> int:
        """Bytes of one sketch step on ``b`` columns."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.eta * self.p * chi * e
        )
        slices = (1 + staged) * b * self.a * e
        peak = (
            peak_elements(
                self.eqs.rtl_m,
                ((b, *self.left), *self.shapes, (self.eta, *self.right)),
            )
            * e
        )
        return fixed + slices + peak

    def qr(self, prefetch: int) -> int:
        """Bytes of the QR of the sketch: the sketch, ``Q`` and a workspace."""
        e = self.e
        return (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + 3 * self.eta * self.p * self.chi * e
        )

    def project(self, b: int, prefetch: int) -> int:
        """Bytes of one projection step on ``b`` rows."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.eta * self.p * chi * e
            + chi * self.a * e
        )
        slices = b * self.eta * self.p * e
        peak = (
            peak_elements(
                self.eqs.rtl_s,
                (
                    (b, self.eta, self.up, self.down),
                    *self.shapes,
                    (self.eta, *self.right),
                ),
            )
            * e
        )
        return fixed + slices + peak

    def first(self, b: int, prefetch: int) -> int:
        """Bytes of one first-site step on ``b`` rows of ``S``."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.a * chi * self.p * e
        )
        peak = peak_elements(self.eqs.first, (*self.shapes, (b, *self.right))) * e
        return fixed + peak


def _largest_batch(cost: Callable[[int], int], limit: int, avail: int) -> int:
    """Return the largest batch in ``[1, limit]`` whose cost fits, or 0.

    Binary search on the evaluated costs; the result is always a batch whose cost
    was checked. Batches of at least `GEMM_MULTIPLE` are rounded down to a multiple
    of it when that still fits.
    """
    if cost(1) > avail:
        return 0
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= avail:
            lo = mid
        else:
            hi = mid - 1
    rounded = lo - lo % GEMM_MULTIPLE
    if lo >= GEMM_MULTIPLE and rounded != lo and cost(rounded) <= avail:
        return rounded
    return lo


class _Planner:
    """Plans one sweep for a fixed prefetch depth."""

    def __init__(self, sites: Sequence[_Site], budgets: Budgets, prefetch: int) -> None:
        self.sites, self.budgets, self.prefetch = sites, budgets, prefetch
        self.n = len(sites)
        self.chi = sites[0].chi
        self.out_total = sum(self.chi * s.p * s.eta * s.e for s in sites)
        # Pinned ring of the site source, plus the host copy being read.
        self.site_ring = (prefetch + 2) * max(s.core_bytes for s in sites)
        # Two batches in and two out; full environments until batches are known.
        self.env_staging = 4 * max(s.env_bytes for s in sites[1:])

    def _costs(self, j: int, tiers: Sequence[Tier]) -> dict[str, Callable[[int], int]]:
        site, pf = self.sites[j], self.prefetch
        costs: dict[str, Callable[[int], int]] = {}
        if j < self.n - 1:
            staged_in = j > 0 and tiers[j] != "device"
            staged_out = tiers[j + 1] != "device"
            costs["env"] = lambda b: site.env(
                b, pf, staged_in=staged_in, staged_out=staged_out
            )
        if j > 0:
            staged = tiers[j] != "device"
            costs["sketch"] = lambda b: site.sketch(b, pf, staged=staged)
            costs["project"] = lambda b: site.project(b, pf)
        else:
            costs["first"] = lambda b: site.first(b, pf)
        return costs

    def _batches(self, tiers: Sequence[Tier], avail: int) -> list[dict[str, int]]:
        batches = []
        for j, site in enumerate(self.sites):
            if j > 0 and site.qr(self.prefetch) > avail:
                _fail(j, "QR", site.qr(self.prefetch), avail)
            chosen = {}
            for kernel, cost in self._costs(j, tiers).items():
                b = _largest_batch(cost, self.chi, avail)
                if b == 0:
                    _fail(j, kernel, cost(1), avail)
                chosen[kernel] = b
            batches.append(chosen)
        return batches

    def _peak(self, tiers: Sequence[Tier], batches: Sequence[dict[str, int]]) -> int:
        return max(
            cost(batches[j][kernel])
            for j in range(self.n)
            for kernel, cost in self._costs(j, tiers).items()
        )

    def _reserved(self, tiers: Sequence[Tier]) -> int:
        """Device bytes held outside the kernels.

        These are the device-tier environments and, on the CPU backend, where the
        device is the host, the output and the staging buffers as well.
        """
        resident = sum(
            s.env_bytes
            for j, s in enumerate(self.sites)
            if j > 0 and tiers[j] == "device"
        )
        if self.budgets.unified:
            spills = any(tier == "disk" for tier in tiers[1:])
            resident += self.out_total + self.site_ring + spills * self.env_staging
        return resident

    def _tiers(self, device_left: int, host_left: int) -> list[Tier]:
        """Assign the newest environments to the fastest tier that holds them."""
        tiers: list[Tier] = ["device"] * self.n
        level: Tier = "device"
        for j in range(self.n - 1, 0, -1):
            need = self.sites[j].env_bytes
            if level == "device" and device_left >= need:
                device_left -= need
                continue
            if level == "device":
                level = "disk" if self.budgets.unified else "host"
            if level == "host" and host_left >= need:
                tiers[j], host_left = "host", host_left - need
                continue
            tiers[j] = level = "disk"
        return tiers

    def plan(self) -> Plan:
        budgets, n = self.budgets, self.n
        staged: list[Tier] = ["device"] + ["disk"] * (n - 1)
        base = budgets.device - self._reserved(staged)
        batches = self._batches(staged, base)
        # Staged batches never grow in the final pass, so they bound the staging.
        self.env_staging = 4 * max(
            max(batches[j].values()) * s.a * s.e for j, s in enumerate(self.sites)
        )

        preferred = [
            {kernel: min(b, PREFERRED_BATCH) for kernel, b in chosen.items()}
            for chosen in batches
        ]
        host_left = budgets.host - self.out_total - self.site_ring - self.env_staging
        tiers = self._tiers(base - self._peak(staged, preferred), host_left)

        disk_bytes = sum(
            s.env_bytes for j, s in enumerate(self.sites) if tiers[j] == "disk"
        )
        if disk_bytes > budgets.disk:
            msg = (
                f"The environments need {disk_bytes} bytes on disk, but only "
                f"{budgets.disk} bytes are free in {budgets.scratch_dir}."
            )
            raise MemoryError(msg)

        reserved = self._reserved(tiers)
        batches = self._batches(tiers, budgets.device - reserved)
        host_env = sum(
            s.env_bytes for j, s in enumerate(self.sites) if tiers[j] == "host"
        )
        host_peak = self.out_total + self.site_ring
        if disk_bytes or host_env:
            host_peak += host_env + self.env_staging
        sites = tuple(
            SitePlan(
                env_batch=chosen.get("env", 0),
                sketch_batch=chosen.get("sketch", 0),
                project_batch=chosen.get("project", chosen.get("first", 0)),
                tier=tiers[j],
            )
            for j, chosen in enumerate(batches)
        )
        return Plan(
            sites,
            self.prefetch,
            device_peak=reserved + self._peak(tiers, batches),
            host_peak=host_peak,
            disk_bytes=disk_bytes,
        )


def _fail(j: int, kernel: str, need: int, avail: int) -> None:
    msg = (
        f"Site {j}: the {kernel} step needs {need} bytes with a batch of one, but "
        f"only {avail} bytes of the device budget are available. The site working "
        "set exceeds the budget."
    )
    raise MemoryError(msg)


def make_plan(
    site_shapes: Sequence[tuple[Shape, ...]],
    site_bytes: Sequence[int],
    chi_out: int,
    dtype: type | np.dtype,
    budgets: Budgets,
) -> Plan:
    """Plan the batches and the environment tiers of one sweep.

    Args:
        site_shapes: For every site, the padded ``(l, r, u, d)`` shape of each layer.
        site_bytes: For every site, the bytes of its cores.
        chi_out: The sketch size.
        dtype: The data type of the computation.
        budgets: The resolved budgets.

    Returns:
        The plan. Prefetching is dropped before giving up.

    Raises:
        MemoryError: If a site does not fit the device budget even with batches of
            one and no prefetching, or the environments do not fit on disk.
    """
    itemsize = np.dtype(dtype).itemsize
    n_sites = len(site_shapes)
    sites = [
        _Site(
            j,
            tuple(shapes),
            core_bytes,
            chi=chi_out,
            itemsize=itemsize,
            n_sites=n_sites,
        )
        for j, (shapes, core_bytes) in enumerate(zip(site_shapes, site_bytes))
    ]
    try:
        return _Planner(sites, budgets, prefetch=1).plan()
    except MemoryError:
        return _Planner(sites, budgets, prefetch=0).plan()
