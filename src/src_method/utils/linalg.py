"""Linear algebra utilities (numpy / cupy compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace

from ._backend import to_numpy

if TYPE_CHECKING:
    from ._backend import Array


def truncated_qr(matrix: Array, cutoff: float) -> Array:
    """QR with SVD-based rank truncation, returning only the isometry.

    Decomposes ``matrix = Q @ R``, then truncates via SVD on R,
    discarding singular values below ``cutoff * sigma_max``.

    When ``cutoff <= 0`` this falls back to a plain QR (no truncation).

    Args:
        matrix: Input matrix of shape (m, n), on any backend.
        cutoff: Relative singular-value threshold.  Singular values
            satisfying ``s < cutoff * s_max`` are discarded.

    Returns:
        The truncated isometry Q of shape ``(m, rank)``, on the device of
        ``matrix``.
    """
    xp = array_namespace(matrix)
    if cutoff <= 0:
        return xp.linalg.qr(matrix)[0]

    transpose = matrix.shape[0] < matrix.shape[1]
    Q, R = xp.linalg.qr(matrix.T if transpose else matrix)
    # R is small: its SVD runs on the host.
    R = to_numpy(R)
    U, S, _ = np.linalg.svd(R.T if transpose else R, full_matrices=False)
    rank = max(1, int((cutoff * S[0] <= S).sum()))
    U = xp.asarray(U[:, :rank])
    return U if transpose else Q @ U
