"""An implementation of the Successive Randomized Compression (SRC) algorithm.

This module includes the functions used for compression of the
following types:

1. MPO randomized compression.
2. MPS randomized compression.

"""

from __future__ import annotations

from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
import structlog
from opt_einsum import contract

from .utils import (
    default_rng,
    get_xp,
    setup_logging,
    to_numpy,
    truncated_qr,
)

if TYPE_CHECKING:
    from types import ModuleType

# Set up logger
setup_logging()
logger = structlog.get_logger(__name__)

# Logger strings
LOG_LTR = "Left-to-right sweep: Computing C tensors..."
LOG_RTL = "Right-to-left sweep: Constructing η tensors..."
LOG_TIME = " - Elapsed time (s)"
LOG_WARN_SMALL = (
    "The current SRC implementation targets tensor networks with 3 or more sites. "
    "Defaulting to the corresponding quimb primitive with SVD."
)

# -----------------------------------------------
# --- Public API --------------------------------
# -----------------------------------------------


def compress(
    tensor: qtn.MatrixProductState | qtn.MatrixProductOperator,
    chi_out: int,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
    seed: int | None = None,
    device: str = "cpu",
) -> qtn.MatrixProductState | qtn.MatrixProductOperator:
    """Applies the Successive Randomized Compression (SRC) algorithm.

    Dispatches to the appropriate implementation based on the types of the input tensors.
    Supported combinations are:

      1. MPS: `tensor` is an MPS. Results in an MPS.
      2. MPO: `tensor` is an MPO. Results in an MPO.

    Args:
        tensor: The tensor network to compress (MPS or MPO).
        chi_out: The desired maximum bond dimension of the output tensor network.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
            When positive, bonds are trimmed to their effective rank by
            discarding singular values below ``cutoff * sigma_max`` at each
            site during the right-to-left sweep.  The SVD operates on the
            small ``(chi_out, chi_out)`` R factor from QR, so overhead is
            minimal.  Set to 0.0 (default) to keep all bonds at chi_out.
        dtype: The data type for the computation.
        seed: An optional seed for the random number generator.
        device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy).  Requires
            the optional ``cupy`` dependency for GPU execution.

    Returns:
        The resulting compressed tensor network (MPS or MPO).

    Raises:
        TypeError: If the input tensor type is unsupported.
        ValueError: If ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.

    """
    xp = get_xp(device)
    prng = default_rng(seed)

    if tensor.nsites < 3:
        logger.warning(LOG_WARN_SMALL)
        compressed_tensor = tensor.copy(deep=True)
        compressed_tensor.compress(max_bond=chi_out, method="svd")
        return compressed_tensor
    if isinstance(tensor, qtn.MatrixProductState):
        return _src_mps(tensor, chi_out, prng, xp, cutoff=cutoff, dtype=dtype)
    if isinstance(tensor, qtn.MatrixProductOperator):
        return _src_mpo(tensor, chi_out, prng, xp, cutoff=cutoff, dtype=dtype)
    msg = f"Unsupported tensor network type: {type(tensor)}"
    raise TypeError(msg)


# -----------------------------------------------
# --- SRC algorithm implementations -------------
# -----------------------------------------------


def _src_mpo(
    mpo: qtn.MatrixProductOperator,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
) -> qtn.MatrixProductOperator:
    """Compress an MPO using the SRC method.

    Args:
        mpo: The MPO to compress as a quimb object.
        chi_out: The desired maximum bond dimension of the output MPO.
        prng: A numpy / cupy random number generator instance.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type for the computation.

    Returns:
        The resulting compressed MPO as quimb object.
    """
    # Problem dimensions
    n_sites = mpo.nsites
    _, phys_up, phys_down = mpo.arrays[0].shape
    logger.info(
        "Starting SRC MPO",
        n_sites=n_sites,
        phys_up=phys_up,
        phys_down=phys_down,
        device=xp.__name__,
    )

    # Views of the tensors (transferred to device once, up front)
    mpo_arrs = [xp.asarray(mpo[i].data) for i in range(n_sites)]

    # ----------------------------------------------
    # --- Left-to-Right Sweep: Compute C tensors ---
    # ----------------------------------------------
    logger.info(LOG_LTR)
    tms = perf_counter_ns()

    omega = xp.asarray(prng.normal(size=(chi_out, phys_up, phys_down))).astype(dtype)
    C = [contract("abc,dbc->ad", omega, mpo_arrs[0])]
    for i in range(1, n_sites - 1):
        omega = xp.asarray(prng.normal(size=(chi_out, phys_up, phys_down))).astype(
            dtype
        )
        C.append(
            contract(
                "ab,acd,becd->ae",
                C[i - 1],
                omega,
                mpo_arrs[i],
            )
        )

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_ltr=tms * 1e-9)

    # -------------------------------------------------
    # --- Right-to-Left Sweep: Construct η tensors ----
    # -------------------------------------------------
    logger.info(LOG_RTL)
    tms = perf_counter_ns()

    eta = [None] * n_sites

    # Last site
    q_last = truncated_qr(
        contract("ab,bcd->acd", C[-1], mpo_arrs[-1])
        .reshape(chi_out, phys_up * phys_down)
        .T,
        cutoff,
        xp,
    )
    chi_right = q_last.shape[1]
    eta[-1] = q_last.T.reshape(chi_right, phys_up, phys_down)
    S = contract("abc,dbc->da", eta[-1].conj(), mpo_arrs[-1])

    # Sites η^(n-1),...,η^(2)
    for j in range(n_sites - 2, 0, -1):
        M = contract("ab,becd,ef->fcda", C[j - 1], mpo_arrs[j], S).reshape(
            chi_right * phys_up * phys_down, chi_out
        )
        Q_trunc = truncated_qr(M, cutoff, xp)
        rank = Q_trunc.shape[1]
        eta[j] = Q_trunc.reshape(chi_right, phys_up, phys_down, rank).transpose(
            3, 0, 1, 2
        )
        S = contract("abcd,fecd,eb->fa", eta[j].conj(), mpo_arrs[j], S)
        C[j - 1] = None
        chi_right = rank

    # First site
    eta[0] = contract("cab,cd->dab", mpo_arrs[0], S)

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_rtl=tms * 1e-9)
    logger.info("SRC MPO complete.")

    return qtn.MatrixProductOperator([to_numpy(t) for t in eta])


def _src_mps(
    mps: qtn.MatrixProductState,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
) -> qtn.MatrixProductState:
    """Compress an MPS using the SRC method.

    Args:
        mps: The MPS to compress as a quimb object.
        chi_out: The desired maximum bond dimension of the output MPS |η>.
        prng: A numpy / cupy random number generator instance.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type for the computation.

    Returns:
        The resulting compressed MPS as a quimb object.
    """
    # Problem dimensions
    n_sites = mps.nsites
    _, phys_dim = mps.arrays[0].shape
    logger.info(
        "Starting SRC MPS", n_sites=n_sites, phys_dim=phys_dim, device=xp.__name__
    )

    # View of the tensors (transferred to device once, up front)
    mps_arrs = [xp.asarray(mps[i].data) for i in range(n_sites)]

    # ----------------------------------------------
    # --- Left-to-Right Sweep: Compute C tensors ---
    # ----------------------------------------------
    logger.info(LOG_LTR)
    tms = perf_counter_ns()

    omega = xp.asarray(prng.normal(size=(chi_out, phys_dim))).astype(dtype)
    C = [contract("ab,cb->ac", omega, mps_arrs[0])]
    for i in range(1, n_sites - 1):
        omega = xp.asarray(prng.normal(size=(chi_out, phys_dim))).astype(dtype)
        C.append(
            contract(
                "ab,ac,bdc->ad",
                C[i - 1],
                omega,
                mps_arrs[i],
            )
        )

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_ltr=tms * 1e-9)

    # -------------------------------------------------
    # --- Right-to-Left Sweep: Construct η tensors ----
    # -------------------------------------------------
    logger.info(LOG_RTL)
    tms = perf_counter_ns()

    eta = [None] * n_sites

    # Last site
    q_last = truncated_qr(
        contract("ab,bc->ac", C[-1], mps_arrs[-1]).T,
        cutoff,
        xp,
    )
    eta[-1] = q_last.T
    S = contract("ab,cb->ca", eta[-1].conj(), mps_arrs[-1])

    # Sites η^(n-1),...,η^(2)
    chi_right = eta[-1].shape[0]
    for j in range(n_sites - 2, 0, -1):
        M = contract("ab,bdc,de->cea", C[j - 1], mps_arrs[j], S).reshape(
            phys_dim * chi_right, chi_out
        )
        Q_trunc = truncated_qr(M, cutoff, xp)
        rank = Q_trunc.shape[1]
        eta[j] = Q_trunc.reshape(phys_dim, chi_right, rank).transpose(2, 1, 0)
        S = contract("acb,deb,ec->da", eta[j].conj(), mps_arrs[j], S)
        C[j - 1] = None
        chi_right = rank

    # First site
    eta[0] = contract("ba,bc->ca", mps_arrs[0], S)

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_rtl=tms * 1e-9)
    logger.info("SRC MPS complete.")

    return qtn.MatrixProductState([to_numpy(t) for t in eta])
