"""GPU (cupy) backend tests.

Skipped automatically when cupy or a CUDA/ROCm device is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import quimb.tensor as qtn

import src_method._sweep as sweep_module
from src_method import Resources, apply, compress, src
from src_method._kernels import SiteKernels
from src_method._plan import (
    GPU_MARGIN_FRACTION,
    GPU_MARGIN_MIN,
    Budgets,
    Plan,
    make_plan,
)
from src_method._sites import padded_shapes, site_bytes

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


# ---------------------------------------
# --- Budgets, batching and spilling ---
# ---------------------------------------


def random_mpo_arrays(bonds: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """Complex Gaussian MPO with the given bonds and physical legs of 2."""

    def site(*shape: int) -> np.ndarray:
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    lefts, rights = [None, *bonds], [*bonds, None]
    return [
        site(*(b for b in (lb, rb) if b is not None), 2, 2)
        for lb, rb in zip(lefts, rights)
    ]


def dense_mpo(train: list[np.ndarray]) -> np.ndarray:
    """Contract an MPO into a ``(U, D)`` matrix."""
    T = train[0]
    for W in train[1:-1]:
        T = np.einsum("aUD,abud->bUuDd", T, W)
        T = T.reshape(W.shape[1], T.shape[1] * T.shape[2], T.shape[3] * T.shape[4])
    T = np.einsum("aUD,aud->UuDd", T, train[-1])
    return T.reshape(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3])


@pytest.fixture
def plans(monkeypatch: pytest.MonkeyPatch) -> list[Plan]:
    """Record the plan of every sweep."""
    recorded: list[Plan] = []

    def spy(*args: Any, **kwargs: Any) -> Plan:
        recorded.append(make_plan(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(sweep_module, "make_plan", spy)
    return recorded


def test_gpu_tiers_and_batches_match_cpu(tmp_path, plans: list[Plan]) -> None:
    """Every tier and small batches on the GPU give the operator of the CPU run."""
    rng = np.random.default_rng(7)
    stack = [random_mpo_arrays([4, 8, 8, 8, 4], rng) for _ in range(4)]
    chi = 64
    shapes, sizes = padded_shapes(stack), site_bytes(stack)
    roomy = make_plan(
        shapes,
        sizes,
        chi,
        np.complex128,
        Budgets(10**10, 10**10, 10**12, tmp_path, unified=False),
    )
    env = chi * 8**4 * 16  # one bulk environment
    tight = Resources(
        gpu_memory=roomy.device_peak // 2,
        host_memory=roomy.host_peak + 5 * env,
        scratch_dir=tmp_path,
    )

    cpu = src(*stack, chi_out=chi, seed=3, dtype=np.complex128)
    gpu = src(
        *stack, chi_out=chi, seed=3, dtype=np.complex128, device="gpu", resources=tight
    )

    assert {site.tier for site in plans[-1].sites} == {"device", "host", "disk"}
    assert min(site.sketch_batch for site in plans[-1].sites[1:]) < chi
    reference = dense_mpo(cpu)
    error = np.linalg.norm(dense_mpo(gpu) - reference) / np.linalg.norm(reference)
    assert error < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_gpu_pool_limit_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap on cupy's pool is lifted after the sweep, also after an error."""
    rng = np.random.default_rng(8)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    pool = cupy.get_default_memory_pool()
    previous = pool.get_limit()

    src(*stack, chi_out=4, seed=0, device="gpu")
    assert pool.get_limit() == previous

    def boom(*_args: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(SiteKernels, "sketch", boom)
    with pytest.raises(RuntimeError, match="boom"):
        src(*stack, chi_out=4, seed=0, device="gpu")
    assert pool.get_limit() == previous


def test_gpu_pool_cap_leaves_room_for_fragmentation(tmp_path) -> None:
    """The pool is capped at the GPU budget plus the margin, not at the budget."""
    rng = np.random.default_rng(9)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    pool = cupy.get_default_memory_pool()
    limits: list[int] = []

    def spy(*_args: object) -> None:
        limits.append(pool.get_limit())
        msg = "spy"
        raise RuntimeError(msg)

    budget = 10**8
    used = pool.used_bytes()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SiteKernels, "env", spy)
        with pytest.raises(RuntimeError, match="spy"):
            src(
                *stack,
                chi_out=4,
                seed=0,
                device="gpu",
                resources=Resources(gpu_memory=budget, scratch_dir=tmp_path),
            )

    _, total = cupy.cuda.runtime.memGetInfo()
    margin = max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
    assert limits == [used + budget + margin]
