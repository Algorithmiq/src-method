"""Relative distances between outputs saved by ``bench_large.py run --save``.

Prints one JSON line per pair: the QR distance ``|a - b| / |a|`` of `bench_large`,
computed in complex128 after promoting both outputs, and whether the two are
bit-identical site by site.

Usage::

    python distances.py A.npz B.npz [C.npz ...]   # every pair against A
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import bench_large  # noqa: E402


def _load(path: Path) -> list[np.ndarray]:
    with np.load(path) as f:
        return [f[k] for k in sorted(f.files)]


def main() -> None:
    """Compare the first output with each of the others."""
    first, *others = map(Path, sys.argv[1:])
    a = _load(first)
    a128 = [x.astype(np.complex128) for x in a]
    for other in others:
        b = _load(other)
        identical = all(
            x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
            for x, y in zip(a, b)
        )
        b128 = [y.astype(np.complex128) for y in b]
        print(
            json.dumps(
                {
                    "a": first.name,
                    "b": other.name,
                    "qr_distance": bench_large._qr_distance(a128, b128),
                    "bit_identical": identical,
                    "dtypes": [str(a[0].dtype), str(b[0].dtype)],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
