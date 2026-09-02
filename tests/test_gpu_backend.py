"""GPU (cupy) backend tests.

Skipped automatically when cupy or a CUDA/ROCm device is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import quimb.tensor as qtn

from src_method import apply, compress

cupy = pytest.importorskip("cupy")


def as_mps(arrays: list[np.ndarray]) -> qtn.MatrixProductState:
    """Wrap a list of site arrays returned by src_method into a quimb MPS."""
    return qtn.MatrixProductState(arrays)


def as_mpo(arrays: list[np.ndarray]) -> qtn.MatrixProductOperator:
    """Wrap a list of site arrays returned by src_method into a quimb MPO."""
    return qtn.MatrixProductOperator(arrays)


@pytest.fixture(autouse=True)
def _require_device() -> None:
    """Skip the whole module when no GPU runtime is reachable."""
    try:
        cupy.cuda.runtime.getDeviceCount()
    except (cupy.cuda.runtime.CUDARuntimeError, RuntimeError) as err:
        pytest.skip(f"No usable GPU device: {err}")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_apply_mpo_mps_gpu_matches_cpu(device: str) -> None:
    """``apply`` MPO @ MPS should produce equivalent compressed states on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    out = as_mps(
        apply(
            H.arrays,
            psi.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H.apply(psi, compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_apply_mpo_mpo_gpu_matches_cpu(device: str) -> None:
    """``apply`` MPO @ MPO should produce equivalent compressed MPOs on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=np.complex128
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H1.apply(H2, compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_compress_mpo_gpu_matches_cpu(device: str) -> None:
    """``compress`` on an MPO should match the CPU result on GPU."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=np.complex128) / 1e8
    C = A + B

    out = as_mpo(
        compress(C.arrays, chi_out=chi_out, dtype=np.complex128, seed=0, device=device)
    )

    np.testing.assert_allclose(C.distance(out), 0.0, atol=1e-6)


def test_apply_mpo_mpo_cpu_gpu_equivalent() -> None:
    """CPU and GPU ``apply`` MPO @ MPO must produce equivalent MPOs."""
    n_sites, phys_dim, chi_out = 8, 4, 32
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    H2 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)

    cpu_out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=42,
            device="cpu",
        )
    )
    gpu_out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=42,
            device="gpu",
        )
    )

    # Gauge freedom means individual tensors can differ; compare the contracted MPOs.
    # atol=1e-6 accounts for floating-point accumulation across different execution orders.
    np.testing.assert_allclose(cpu_out.distance(gpu_out), 0.0, atol=1e-6)


def test_compress_mpo_cpu_gpu_equivalent() -> None:
    """CPU and GPU ``compress`` must produce equivalent MPOs."""
    n_sites, phys_dim, chi_out = 8, 4, 32
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)

    cpu_out = as_mpo(
        compress(A.arrays, chi_out=chi_out, dtype=np.complex128, seed=42, device="cpu")
    )
    gpu_out = as_mpo(
        compress(A.arrays, chi_out=chi_out, dtype=np.complex128, seed=42, device="gpu")
    )

    np.testing.assert_allclose(cpu_out.distance(gpu_out), 0.0, atol=1e-6)


def test_invalid_device_raises() -> None:
    """An unrecognised ``device`` must raise ``ValueError``."""
    H = qtn.MPO_rand(5, bond_dim=4, phys_dim=2, dtype=np.complex128)
    psi = qtn.MPS_rand_state(5, bond_dim=4, phys_dim=2, dtype=np.complex128)
    with pytest.raises(ValueError, match="Unknown device"):
        apply(H.arrays, psi.arrays, chi_out=4, device="tpu")
