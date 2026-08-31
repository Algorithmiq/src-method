"""Benchmarking the MPO-MPO contraction-compression."""

from __future__ import annotations

from time import perf_counter_ns

import cyclopts
import numpy as np
import quimb.tensor as qtn
import structlog

from src_method import apply

# Set up logger
logger = structlog.get_logger()


# Initialize cyclopts app
app = cyclopts.App(help="Run SRC benchmark.")


@app.default
def main(
    n_sites: int = 50,
    chi_out: int = 20,
    run: str = "src",
    compare: str = "yes",
) -> None:
    """Main benchmarking function.

    Args:
        n_sites: Number of sites in the MPO chain.
        chi_out: Output bond dimension for compression.
        run: Which library to run ('quimb', 'src').
        compare: Whether to compare results to a reference ('yes', 'no').
    """
    phys_dim = 2
    array_type = np.complex128

    logger.info(
        "benchmark_start",
        n_sites=n_sites,
        chi_out=chi_out,
        phys_dim=phys_dim,
        dtype="complex128",
        run=run,
        compare=compare,
    )

    # Generate a random MPO and a perturbed identity MPO
    logger.info("Generating MPOs...")
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=3, phys_dim=phys_dim, dtype=array_type
    )  # Total bond dimension is 4: a near-identity, low-rank operator

    # Quimb's contraction reference
    if compare == "yes":
        logger.info("Computing reference contraction (no compression)...")
        tms = perf_counter_ns()
        H_ref = H1.apply(H2, compress=False)
        tms = perf_counter_ns() - tms
        logger.info("Reference contraction took %s s", tms * 1e-9)

    # Quimb's contraction with compression
    if run == "quimb":
        logger.info("Computing Quimb's MPO-MPO contraction (with compression)...")
        tms = perf_counter_ns()
        H_quimb = H1.apply(H2, compress=True, max_bond=chi_out, method="rsvd")
        tms = perf_counter_ns() - tms
        logger.info(" Quimb's contraction-compression took %s s", tms * 1e-9)
        if compare == "yes":
            logger.info(" - Distance to reference: %s", H_ref.distance(H_quimb))

    # SRC's contraction compression
    if run == "src":
        logger.info("Computing SRC's MPO-MPO contraction (with compression)...")
        tms = perf_counter_ns()
        H_src = apply(H1, H2, chi_out=chi_out, dtype=array_type)
        tms = perf_counter_ns() - tms
        logger.info(" SRC's contraction-compression took %s s", tms * 1e-9)
        if compare == "yes":
            logger.info(" - Distance to reference: %s", H_ref.distance(H_src))

    logger.info("benchmark_end")


if __name__ == "__main__":
    app()
