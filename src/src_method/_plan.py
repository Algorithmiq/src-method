"""Memory planning for the SRC sweep: budgets, batch sizes and environment tiers."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import device_memory, host_memory_available, is_host

if TYPE_CHECKING:
    import os
    from types import ModuleType

# Kept free on the device for cuBLAS/cuSOLVER workspaces and pool fragmentation.
GPU_MARGIN_FRACTION = 0.10
GPU_MARGIN_MIN = 2**30
HOST_MARGIN_FRACTION = 0.10
DISK_MARGIN_FRACTION = 0.05

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
        gpu_memory: Device memory the sweep may use, as a byte count or a size
            string (``"36GB"``, ``"36GiB"``). Defaults to the free device memory
            minus ``max(10%, 1 GiB)``. Ignored on the CPU.
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
    """

    device: int
    host: int
    disk: int
    scratch_dir: Path
    unified: bool


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
    if unified:
        device = host
    elif resources.gpu_memory is not None:
        device = parse_size(resources.gpu_memory)
    else:
        available, total = device_memory(xp)
        device = available - max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
    free = shutil.disk_usage(_existing_parent(scratch)).free
    disk = int(free * (1 - DISK_MARGIN_FRACTION))
    return Budgets(max(device, 0), max(host, 0), disk, scratch, unified)


def _existing_parent(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path
