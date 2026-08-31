"""An implementation of the Successive Randomized Compression (SRC) algorithm.

This module includes the functions used for contraction-compressions of the
following types:

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.

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


def apply(
    left_tensor: qtn.MatrixProductOperator,
    right_tensor: qtn.MatrixProductOperator | qtn.MatrixProductState,
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

      1. MPO-MPS: `left_tensor` is an MPO and `right_tensor` is an MPS. Results in an MPS.
      2. MPO-MPO: both `left_tensor` and `right_tensor` are MPOs. Results in an MPO.

    Args:
        left_tensor: The left tensor network (MPO).
        right_tensor: The right tensor network (MPO or MPS).
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
        TypeError: If the combination of input tensor types is unsupported.
        ValueError: If ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    xp = get_xp(device)
    prng = default_rng(seed)

    if left_tensor.nsites < 3:
        logger.warning(LOG_WARN_SMALL)
        return left_tensor.apply(
            right_tensor, compress=True, max_bond=chi_out, method="svd"
        )
    if isinstance(left_tensor, qtn.MatrixProductOperator) and isinstance(
        right_tensor, qtn.MatrixProductState
    ):
        return _src_mpo_mps(
            left_tensor, right_tensor, chi_out, prng, xp, cutoff=cutoff, dtype=dtype
        )
    if isinstance(left_tensor, qtn.MatrixProductOperator) and isinstance(
        right_tensor, qtn.MatrixProductOperator
    ):
        return _src_mpo_mpo(
            left_tensor, right_tensor, chi_out, prng, xp, cutoff=cutoff, dtype=dtype
        )
    msg = (
        "Unsupported combination of tensor network types: "
        f"{type(left_tensor)} and {type(right_tensor)}"
    )
    raise TypeError(msg)


# -----------------------------------------------
# --- SRC algorithm implementations -------------
# -----------------------------------------------


def _src_mpo_mps(
    mpo: qtn.MatrixProductOperator,
    mps: qtn.MatrixProductState,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
) -> qtn.MatrixProductState:
    """Computes the compressed product |η> ≈ H|ψ> using the SRC method.

    Args:
        mpo: The MPO, as quimb object.
        mps: The MPS, as a quimb object.
        chi_out: The desired maximum bond dimension of the output MPS |η>.
        prng: A numpy / cupy random number generator.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type for the computation.

    Returns:
        The resulting compressed MPS |η> in right-canonical form.
    """
    # Problem dimensions
    n_sites = mpo.nsites
    phys_dim = mps[0].ind_size(mps.site_ind(0))
    logger.info(
        "Starting SRC MPO-MPS", n_sites=n_sites, phys_dim=phys_dim, device=xp.__name__
    )

    # Views of the tensors (transferred to device once, up front)
    mpo_arrs = [xp.asarray(mpo[i].data) for i in range(n_sites)]
    mps_arrs = [xp.asarray(mps[i].data) for i in range(n_sites)]

    # ----------------------------------------------
    # --- Left-to-Right Sweep: Compute C tensors ---
    # ----------------------------------------------
    logger.info(LOG_LTR)
    tms = perf_counter_ns()

    omega = xp.asarray(prng.normal(size=(phys_dim, chi_out))).astype(dtype)
    C = [contract("da,bde,ce->abc", omega, mpo_arrs[0], mps_arrs[0])]
    for i in range(1, n_sites - 1):
        omega = xp.asarray(prng.normal(size=(phys_dim, chi_out))).astype(dtype)
        C.append(
            contract(
                "ade,fa,dbfg,ecg->abc",
                C[i - 1],
                omega,
                mpo_arrs[i],
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
        contract("acd,cbe,de->ab", C[-1], mpo_arrs[-1], mps_arrs[-1]).T,
        cutoff,
        xp,
    )
    eta[-1] = q_last.T
    S = contract(
        "ad,bde,ce->abc",
        eta[-1].conj(),
        mpo_arrs[-1],
        mps_arrs[-1],
    )

    # Sites η^(n-1),...,η^(2)
    chi_right = eta[-1].shape[0]
    for j in range(n_sites - 2, 0, -1):
        M = contract(
            "abc,bdfg,ceg,hde->fha",
            C[j - 1],
            mpo_arrs[j],
            mps_arrs[j],
            S,
        ).reshape(phys_dim * chi_right, chi_out)
        Q_trunc = truncated_qr(M, cutoff, xp)
        rank = Q_trunc.shape[1]
        eta[j] = Q_trunc.reshape(phys_dim, chi_right, rank).transpose(2, 1, 0)
        S = contract(
            "acb,debf,ghf,ceh->adg",
            eta[j].conj(),
            mpo_arrs[j],
            mps_arrs[j],
            S,
        )
        C[j - 1] = None
        chi_right = rank

    # First site
    eta[0] = contract("bac,dc,ebd->ea", mpo_arrs[0], mps_arrs[0], S)

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_rtl=tms * 1e-9)
    logger.info("SRC MPO-MPS complete.")

    return qtn.MatrixProductState([to_numpy(t) for t in eta])


def _src_mpo_mpo(
    mpo_left: qtn.MatrixProductOperator,
    mpo_right: qtn.MatrixProductOperator,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
) -> qtn.MatrixProductOperator:
    """Computes the compressed product H_new ≈ H1 @ H2 using the SRC method.

    Args:
        mpo_left: The first MPO.
        mpo_right: The second MPO.
        chi_out: The desired maximum bond dimension of the output MPO.
        prng: A numpy / cupy random number generator.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type for the computation.

    Returns:
        The resulting compressed MPO in right-canonical form.
    """
    # Problem dimensions
    n_sites = mpo_left.nsites
    phys_up, phys_down = mpo_left.arrays[0].shape[1], mpo_right.arrays[0].shape[2]
    logger.info(
        "Starting SRC MPO-MPO",
        n_sites=n_sites,
        phys_up=phys_up,
        phys_down=phys_down,
        device=xp.__name__,
    )

    # Views of the tensors (transferred to device once, up front)
    mpo_left_arrs = [xp.asarray(mpo_left[i].data) for i in range(n_sites)]
    mpo_right_arrs = [xp.asarray(mpo_right[i].data) for i in range(n_sites)]

    # ----------------------------------------------
    # --- Left-to-Right Sweep: Compute C tensors ---
    # ----------------------------------------------
    logger.info(LOG_LTR)
    tms = perf_counter_ns()

    omega = xp.asarray(prng.normal(size=(chi_out, phys_up, phys_down))).astype(dtype)
    C = [contract("abc,ebd,fdc->aef", omega, mpo_left_arrs[0], mpo_right_arrs[0])]

    for i in range(1, n_sites - 1):
        omega = xp.asarray(prng.normal(size=(chi_out, phys_up, phys_down))).astype(
            dtype
        )
        C.append(
            contract(
                "abc,ade,bfdh,cghe->afg",
                C[i - 1],
                omega,
                mpo_left_arrs[i],
                mpo_right_arrs[i],
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
        contract("abc,bde,cef->adf", C[-1], mpo_left_arrs[-1], mpo_right_arrs[-1])
        .reshape(chi_out, phys_up * phys_down)
        .T,
        cutoff,
        xp,
    )
    chi_right = q_last.shape[1]
    eta[-1] = q_last.T.reshape(chi_right, phys_up, phys_down)
    S = contract(
        "abc,dbe,fec->adf", eta[-1].conj(), mpo_left_arrs[-1], mpo_right_arrs[-1]
    )

    # Sites η^(n-1),...,η^(2)
    for j in range(n_sites - 2, 0, -1):
        M = contract(
            "abc,bdfg,cegh,ide->ifha",
            C[j - 1],
            mpo_left_arrs[j],
            mpo_right_arrs[j],
            S,
        ).reshape(chi_right * phys_up * phys_down, chi_out)
        Q_trunc = truncated_qr(M, cutoff, xp)
        rank = Q_trunc.shape[1]
        eta[j] = Q_trunc.reshape(chi_right, phys_up, phys_down, rank).transpose(
            3, 0, 1, 2
        )
        S = contract(
            "abcd,hecg,ifgd,bef->ahi",
            eta[j].conj(),
            mpo_left_arrs[j],
            mpo_right_arrs[j],
            S,
        )
        C[j - 1] = None
        chi_right = rank

    # First site
    eta[0] = contract("bac,dce,fbd->fae", mpo_left_arrs[0], mpo_right_arrs[0], S)

    tms = perf_counter_ns() - tms
    logger.debug(LOG_TIME, t_rtl=tms * 1e-9)
    logger.info("SRC MPO-MPO complete.")

    return qtn.MatrixProductOperator([to_numpy(t) for t in eta])
