"""GPU (cupy) backend tests.

Skipped automatically when cupy or a CUDA/ROCm device is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import quimb.tensor as qtn

from src_method import apply, compress, src

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


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_src_stack_gpu_matches_reference(device: str) -> None:
    """A depth-3 ``src`` stack should match the exact product on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 16
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=2, phys_dim=phys_dim, dtype=np.complex128, seed=1
    )
    H2 = qtn.MPO_rand(
        n_sites, bond_dim=2, phys_dim=phys_dim, dtype=np.complex128, seed=2
    )
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=3
    )

    out = as_mps(
        src(
            H1.arrays,
            H2.arrays,
            psi.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H1.apply(H2.apply(psi, compress=False), compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


def test_apply_mpo_mps_to_host_false_returns_cupy() -> None:
    """With to_host=False, apply() must return cupy arrays on GPU."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    result = apply(
        H.arrays,
        psi.arrays,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="gpu",
        to_host=False,
    )
    assert all(isinstance(t, cupy.ndarray) for t in result)


def test_compress_mpo_to_host_false_returns_cupy() -> None:
    """With to_host=False, compress() must return cupy arrays on GPU."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)

    result = compress(
        H.arrays,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="gpu",
        to_host=False,
    )
    assert all(isinstance(t, cupy.ndarray) for t in result)


def test_chained_apply_no_host_roundtrip() -> None:
    """Chained GPU calls with to_host=False must work and match CPU results."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H1 = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)
    H2 = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    # 1. Chained GPU apply
    intermediate_gpu = apply(
        H1.arrays,
        psi.arrays,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="gpu",
        to_host=False,
    )
    assert isinstance(intermediate_gpu[0], cupy.ndarray)

    final_gpu = apply(
        H2.arrays,
        intermediate_gpu,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="gpu",
        to_host=False,
    )
    assert isinstance(final_gpu[0], cupy.ndarray)

    # 2. Chained CPU apply for numerical comparison
    intermediate_cpu = apply(
        H1.arrays,
        psi.arrays,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="cpu",
        to_host=True,
    )
    final_cpu = apply(
        H2.arrays,
        intermediate_cpu,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="cpu",
        to_host=True,
    )

    # Bring GPU result to host and compare
    final_gpu_host = [t.get() for t in final_gpu]

    qtn_cpu = as_mps(final_cpu)
    qtn_gpu = as_mps(final_gpu_host)
    np.testing.assert_allclose(qtn_cpu.distance(qtn_gpu), 0.0, atol=1e-6)


def test_to_host_true_backward_compatible() -> None:
    """to_host=True (implicit default) must return numpy arrays even when device='gpu'."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)

    result = compress(
        H.arrays,
        chi_out=chi_out,
        dtype=np.complex128,
        seed=0,
        device="gpu",
        # to_host=True is the implicit default here
    )
    assert all(isinstance(t, np.ndarray) for t in result)
