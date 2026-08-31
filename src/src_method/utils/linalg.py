"""Linear algebra utilities (numpy / cupy compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


def truncated_qr(matrix: NDArray, cutoff: float, xp: ModuleType = np) -> NDArray:
    """QR with SVD-based rank truncation, returning only the isometry.

    Decomposes ``matrix = Q @ R``, then truncates via SVD on R,
    discarding singular values below ``cutoff * sigma_max``.

    When ``cutoff <= 0`` this falls back to a plain QR (no truncation).

    Args:
        matrix: Input matrix of shape (m, n).
        cutoff: Relative singular-value threshold.  Singular values
            satisfying ``s < cutoff * s_max`` are discarded.
        xp: Array module (``numpy`` or ``cupy``); defaults to numpy.

    Returns:
        The truncated isometry Q of shape ``(m, rank)``.
    """
    m, n = matrix.shape
    transpose = m < n

    if cutoff <= 0:
        return xp.linalg.qr(matrix)[0]

    Q, R = xp.linalg.qr(matrix.T if transpose else matrix)
    R_np = R.get() if hasattr(R, "get") else np.asarray(R)
    U, S, _ = np.linalg.svd(R_np.T if transpose else R_np, full_matrices=False)
    rank = max(1, int((cutoff * S[0] <= S).sum()))

    if transpose:
        Q_trunc = U[:, :rank]
    else:
        U_trunc = xp.asarray(U[:, :rank]) if xp is not np else U[:, :rank]
        Q_trunc = Q @ U_trunc

    if xp is not np:
        return xp.asarray(Q_trunc)
    return Q_trunc
