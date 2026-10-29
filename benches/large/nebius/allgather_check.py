"""Time `DeviceGroup.allgather` alone, with the payloads of the reference sweep.

Isolates the collective from the sweep: every rank publishes a block of
``total / G`` bytes and pulls the others, as for ``S_j`` (4.096 GB) and the cores
(2.048 GB). Prints one JSON line per ``G`` and payload with the per-receiver rate,
to compare with the ``allgather_gbs`` probe, which issues the same peer copies
without the collective's events and barriers.

Usage::

    python allgather_check.py [--reps 5]
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter

import cupy as cp

from src_method._group import DeviceGroup

GB = 1e9


def main() -> None:
    """Time the allgather at G = 2, 4, 8 for both payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    for total in (4.096e9, 2.048e9):
        for g in (2, 4, 8):
            group = DeviceGroup("gpu", g)
            rows = int(total / 8 / g / 1024)

            def fn(rank: int, group: DeviceGroup = group, rows: int = rows) -> float:
                local = cp.ones((rows, 1024), dtype=cp.complex64)
                times = []
                for _ in range(args.reps + 1):
                    group._wait()  # noqa: SLF001
                    cp.cuda.Device().synchronize()
                    group._wait()  # noqa: SLF001
                    start = perf_counter()
                    full = group.allgather(rank, local)
                    cp.cuda.Device().synchronize()
                    group._wait()  # noqa: SLF001
                    times.append(perf_counter() - start)
                    del full
                return median(times[1:])

            seconds = max(group.run(fn))
            per_receiver = (g - 1) / g * total
            print(
                json.dumps(
                    {
                        "total_bytes": total,
                        "gpus": g,
                        "seconds": seconds,
                        "per_receiver_gbs": per_receiver / seconds / GB,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
