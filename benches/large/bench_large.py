"""Out-of-core SRC of ``N . V . M . U`` with a large ``M``, read from disk.

Three commands:

- ``generate``: write the four MPOs site by site as ``.npy`` files, so that ``M``
  is never whole in memory.
- ``run``: compress the stack with automatic (or given) budgets and report the
  plan, the wall time, the device pool size, the host peak and the stall time.
- ``compare``: run twice with two GPU budgets and report the relative distance
  between the two outputs, which checks that the plan does not change the result.

The plan, the per-pass times and the stall time are also logged by `src_method`
itself at ``DEBUG``; pass ``--debug`` to see them.
"""

from __future__ import annotations

import logging
import resource
from pathlib import Path  # noqa: TC003  (cyclopts reads the annotations at runtime)
from time import perf_counter
from typing import Annotated

import cyclopts
import numpy as np

import src_method._sweep as sweep_module
from src_method import Resources, src
from src_method._plan import make_plan

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Benchmark out-of-core SRC of N.V.M.U with a large M.")

LAYERS = ("N", "V", "M", "U")
PHYS = 4  # Pauli transfer matrix legs


def _site_shape(j: int, n_sites: int, bond: int) -> tuple[int, ...]:
    left = () if j == 0 else (bond,)
    right = () if j == n_sites - 1 else (bond,)
    return (*left, *right, PHYS, PHYS)


@app.command
def generate(
    directory: Path,
    *,
    n_sites: int = 50,
    bond: int = 4,
    bond_m: int = 4000,
    dtype: str = "complex128",
    seed: int = 0,
) -> None:
    """Write random MPOs ``N``, ``V``, ``M``, ``U`` as one ``.npy`` file per site.

    Args:
        directory: Where to write, ideally node-local NVMe.
        n_sites: Number of sites.
        bond: Bond dimension of ``N``, ``V`` and ``U``.
        bond_m: Bond dimension of ``M``.
        dtype: ``float64`` or ``complex128``.
        seed: Seed of the random draws.
    """
    rng = np.random.default_rng(seed)
    kind = np.dtype(dtype)
    for name in LAYERS:
        chi = bond_m if name == "M" else bond
        (directory / name).mkdir(parents=True, exist_ok=True)
        # Scaled so that products of the layers stay of order one.
        scale = 1 / np.sqrt(chi * PHYS)
        for j in range(n_sites):
            shape = _site_shape(j, n_sites, chi)
            site = np.lib.format.open_memmap(
                directory / name / f"{j:04d}.npy", mode="w+", dtype=kind, shape=shape
            )
            for row in range(shape[0]):  # one left-bond slice at a time
                draw = rng.normal(size=shape[1:])
                if kind.kind == "c":
                    draw = draw + 1j * rng.normal(size=shape[1:])
                site[row] = draw * scale
            site.flush()
            del site
        logger.info("Layer %s written: bond %d", name, chi)


def _load(directory: Path) -> list[list[np.ndarray]]:
    return [
        [
            np.load(path, mmap_mode="r")
            for path in sorted((directory / name).glob("*.npy"))
        ]
        for name in LAYERS
    ]


def _run(
    directory: Path, chi_out: int, resources: Resources, seed: int
) -> list[np.ndarray]:
    layers = _load(directory)
    plans = []

    def spy(*args: object, **kwargs: object) -> object:
        plans.append(make_plan(*args, **kwargs))
        return plans[-1]

    sweep_module.make_plan = spy  # record the plan of the run
    try:
        start = perf_counter()
        out = src(
            *layers,
            chi_out=chi_out,
            dtype=layers[0][0].dtype,
            seed=seed,
            device="gpu",
            resources=resources,
        )
        seconds = perf_counter() - start
    finally:
        sweep_module.make_plan = make_plan
    import cupy  # noqa: PLC0415  (GPU-only benchmark)

    (plan,) = plans
    tiers = [site.tier for site in plan.sites]
    logger.info(
        "Run complete: %.1f s, pool %d B, host peak %d B, planned device peak %d B, "
        "planned host peak %d B, disk %d B, tiers %s, sketch batches %s",
        seconds,
        cupy.get_default_memory_pool().total_bytes(),
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        plan.device_peak,
        plan.host_peak,
        plan.disk_bytes,
        {tier: tiers.count(tier) for tier in ("device", "host", "disk")},
        sorted({site.sketch_batch for site in plan.sites[1:]}),
    )
    return out


@app.command
def run(
    directory: Path,
    *,
    chi_out: int = 2000,
    gpu_memory: str | None = None,
    host_memory: str | None = None,
    scratch_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Compress the stack written by ``generate`` on the GPU.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        gpu_memory: GPU budget, e.g. ``36GB``; detected when omitted.
        host_memory: Host budget; detected when omitted.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        seed: Seed of the sketch.
    """
    _run(directory, chi_out, Resources(gpu_memory, host_memory, scratch_dir), seed)


def _inner(a: list[np.ndarray], b: list[np.ndarray]) -> complex:
    """Frobenius inner product ``<a, b>`` of two MPOs, site by site."""
    env = np.einsum("rud,sud->rs", a[0].conj(), b[0])
    for x, y in zip(a[1:-1], b[1:-1]):
        env = np.einsum("rs,rtud,svud->tv", env, x.conj(), y, optimize=True)
    return complex(np.einsum("rs,rud,sud->", env, a[-1].conj(), b[-1]))


@app.command
def compare(
    directory: Path,
    *,
    chi_out: int = 500,
    small: str = "40GB",
    large: str = "80GB",
    scratch_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Run with two GPU budgets and report the relative distance of the outputs.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        small: The smaller GPU budget.
        large: The larger GPU budget.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        seed: Seed of the sketch, the same for both runs.
    """
    first = _run(directory, chi_out, Resources(small, None, scratch_dir), seed)
    second = _run(directory, chi_out, Resources(large, None, scratch_dir), seed)
    aa, bb, ab = _inner(first, first), _inner(second, second), _inner(first, second)
    distance = np.sqrt(max((aa + bb - 2 * ab).real, 0.0) / aa.real)
    logger.info("Relative distance between the runs: %.3e", distance)


@app.meta.default
def main(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    debug: bool = False,
) -> None:
    """Configure logging, then dispatch to a command.

    Args:
        tokens: The command and its arguments.
        debug: Also show the `src_method` debug log: plan, pass times and stalls.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if debug:
        logging.getLogger("src_method").setLevel(logging.DEBUG)
    app(tokens)


if __name__ == "__main__":
    app.meta()
