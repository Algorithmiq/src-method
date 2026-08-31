"""Test the four flavors of the SRC algorithm.

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.
3. MPO randomized compression.
4. MPS randomized compression.


"""

import numpy as np
import pytest
import quimb.tensor as qtn

from src_method import apply, compress

# -------------
# --- Utils ---
# -------------


def random_mpo(
    bonds: list[int],
    *,
    phys: int = 2,
    dtype: type = np.complex128,
    rng: np.random.Generator = np.random.default_rng(),
) -> qtn.MatrixProductOperator:
    """Generate a random MPO with given bond dimensions and physical dimension.

    Useful for generating jagged MPOs.

    Args:
        bonds: List of bond dimensions for the MPO. The last site is implicit.
        phys: Physical dimension. Defaults to 2.
        dtype: Data type of the tensors. Defaults to np.complex128.
        rng: Random number generator. Defaults to np.random.default_rng().

    Returns:
        qtn.MatrixProductOperator: The generated random MPO.
    """
    # First site
    tensors = [rng.normal(size=(bonds[0], phys, phys)).astype(dtype)]

    # Bulk sites
    tensors.extend(
        rng.normal(size=(bonds[i - 1], bonds[i], phys, phys)).astype(dtype)
        for i in range(1, len(bonds))
    )
    # Last site
    tensors.append(rng.normal(size=(bonds[-1], phys, phys)).astype(dtype))

    return qtn.MatrixProductOperator(tensors)


# ----------------
# --- Fixtures ---
# ----------------


@pytest.fixture
def n_sites():
    return 5


@pytest.fixture
def n_sites_quimb():
    return 2


@pytest.fixture
def phys_dim():
    return 2


@pytest.fixture
def chi_out():
    return 10


@pytest.fixture
def array_type():
    return np.complex128


@pytest.fixture
def mpo_jagged_left(phys_dim, array_type):
    return random_mpo([1, 4, 1, 4, 1], phys=phys_dim, dtype=array_type) * 1e-8


@pytest.fixture
def mpo_jagged_right(phys_dim, array_type):
    return random_mpo([100, 400, 100, 400, 100], phys=phys_dim, dtype=array_type)


# --------------------------------------------
# --- Test MPO-MPS contraction-compression ---
# --------------------------------------------


def test_src_mpo_mps(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPS contraction-compression."""

    # Generate a random MPS and connect it to the identity MPO
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # SRC MPS should be identical to the original
    psi_compress = apply(H, psi, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(psi.distance(psi_compress), 0.0, atol=1e-6)


def test_src_mpo_mps_trims_terminal_bond() -> None:
    """MPO-MPS apply should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 2, 64
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=3
    )

    psi_src = apply(H, psi, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert psi_src.arrays[-1].shape[0] < chi_out
    assert psi_src.arrays[-1].shape[0] <= phys_dim


# --------------------------------------------
# --- Test MPO-MPO contraction-compression ---
# --------------------------------------------


def test_src_mpo_mpo_identity(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""

    # Generate a random MPO and the identity MPO
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)

    # The compressed product should be identical to the original MPO
    H_compress = apply(H1, H2, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(H1.distance(H_compress), 0.0, atol=1e-6)


def test_src_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""

    # Generate a random MPO and a perturbed identity MPO
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Quimb's contraction
    H_ref = H1.apply(H2, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = apply(H1, H2, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_long_phys(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""
    phys_dim_long = 3 * phys_dim  # triggers transpose in isometry

    # Generate a random MPO and a perturbed identity MPO
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim_long, dtype=array_type
    )
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim_long, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim_long, dtype=array_type
    )

    # Quimb's contraction
    H_ref = H1.apply(H2, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = apply(H1, H2, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_jagged(mpo_jagged_left, mpo_jagged_right, array_type):
    """Tests the SRC MPO-MPO contraction-compression with jagged MPOs."""
    # Quimb's contraction
    H_ref = mpo_jagged_left.apply(mpo_jagged_right, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = apply(mpo_jagged_left, mpo_jagged_right, chi_out=100, dtype=array_type)

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_trims_terminal_bond() -> None:
    """MPO-MPO apply should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 4, 128
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=1
    )
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=np.complex128
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=2
    )

    H_src = apply(H1, H2, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert H_src.arrays[-1].shape[0] < chi_out
    assert H_src.arrays[-1].shape[0] <= phys_dim**2


# --------------------------------
# ----- Test MPO compression -----
# --------------------------------


def test_src_mpo_compression(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO compression."""

    # Add an almost-zero MPO B to a dense MPO A, inflating the bond dimension
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type) / 1e8
    C = A + B

    # SRC MPO. It should be trivially compressed.
    D = compress(C, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_src_mpo_compression_trims_terminal_bond() -> None:
    """MPO compression should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 4, 128
    A = qtn.MPO_rand(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=4
    )
    B = (
        qtn.MPO_rand(
            n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=5
        )
        / 1e8
    )
    C = A + B

    D = compress(C, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert D.arrays[-1].shape[0] < chi_out
    assert D.arrays[-1].shape[0] <= phys_dim**2


# --------------------------------
# ----- Test MPS compression -----
# --------------------------------


def test_src_mps(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPS compression."""

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = compress(C, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_src_mps_compression_trims_terminal_bond() -> None:
    """MPS compression should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 2, 64
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=6
    )
    B = (
        qtn.MPS_rand_state(
            n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=7
        )
        / 1e8
    )
    C = A + B

    D = compress(C, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert D.arrays[-1].shape[0] < chi_out
    assert D.arrays[-1].shape[0] <= phys_dim


def test_src_mps_small_chi(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPS compression with a smaller BD.

    Checks that the code works when wide matrices are involved.
    """
    chi_small = chi_out // 2

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = compress(C, chi_out=chi_small, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


# -----------------------------------------------
# --- Test default to quimb for small systems ---
# -----------------------------------------------


def test_apply_quimb_dispatch(n_sites_quimb, phys_dim, chi_out, array_type):
    """Tests that apply dispatches to quimb for small systems."""

    # Generate a random MPS and connect it to the identity MPO
    H = qtn.MPO_identity(n_sites_quimb, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites_quimb, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # SRC MPS should be identical to the original
    psi_compress = apply(H, psi, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(psi.distance(psi_compress), 0.0, atol=1e-6)


def test_compress_quimb_dispatch(n_sites_quimb, phys_dim, chi_out, array_type):
    """Tests that compress dispatches to quimb for small systems."""

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites_quimb, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(
            n_sites_quimb, bond_dim=5, phys_dim=phys_dim, dtype=array_type
        )
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = compress(C, chi_out=chi_out, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


# -------------------------------------------------
# --- Test adaptive bond truncation via cutoff  ---
# -------------------------------------------------


def test_cutoff_zero_matches_default_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """cutoff=0 should produce the same result as no cutoff (default)."""
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    H_default = apply(H1, H2, chi_out=chi_out, dtype=array_type, seed=42)
    H_cutoff0 = apply(H1, H2, chi_out=chi_out, cutoff=0.0, dtype=array_type, seed=42)

    np.testing.assert_allclose(H_default.distance(H_cutoff0), 0.0, atol=1e-6)


def test_cutoff_trims_bonds_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """With cutoff>0, bonds with low effective rank should be trimmed."""
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)

    # Identity product: effective rank = chi_out, no truncation expected
    H_no_cut = apply(H1, H2, chi_out=chi_out, dtype=array_type, seed=42)
    H_cut = apply(H1, H2, chi_out=chi_out, cutoff=1e-10, dtype=array_type, seed=42)

    # Result should still be accurate
    np.testing.assert_allclose(H1.distance(H_cut), 0.0, atol=1e-5)

    # Bonds should be <= those without cutoff
    for b_cut, b_nocut in zip(H_cut.bond_sizes(), H_no_cut.bond_sizes()):
        assert b_cut <= b_nocut


def test_cutoff_preserves_accuracy_mpo_mps(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPO-MPS."""
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    psi_cut = apply(H, psi, chi_out=chi_out, cutoff=1e-10, dtype=array_type)

    np.testing.assert_allclose(psi.distance(psi_cut), 0.0, atol=1e-6)


def test_cutoff_preserves_accuracy_compress_mpo(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPO compression."""
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type) / 1e8
    C = A + B

    D = compress(C, chi_out=chi_out, cutoff=1e-10, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_cutoff_preserves_accuracy_compress_mps(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPS compression."""
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    D = compress(C, chi_out=chi_out, cutoff=1e-10, dtype=array_type)

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


# --------------------------------------------
# --- Test unsupported tensor combinations ---
# --------------------------------------------


def test_apply_unsupported_types(n_sites, phys_dim, chi_out, array_type):
    """Tests that apply raises TypeError for unsupported tensor combinations."""

    # Generate a random MPS and MPO
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    phi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Unsupported combination: MPS-MPS
    with pytest.raises(TypeError):
        apply(psi, phi, chi_out=chi_out, dtype=array_type)


def test_compress_unsupported_type(n_sites, chi_out, array_type):
    """Tests that compress raises TypeError for unsupported tensor types."""

    # Generate a random PEPS, which is unsupported
    peps = qtn.PEPS.rand(Lx=n_sites, Ly=n_sites, bond_dim=chi_out)

    # Unsupported type: PEPS
    with pytest.raises(TypeError):
        compress(peps, chi_out=chi_out, dtype=array_type)


# -----------------------------------------
# --- Check for performance regressions ---
# -----------------------------------------


@pytest.mark.perf
def test_benchmark_src_mpo_mpo(benchmark):
    """Benchmarks the SRC MPO-MPO contraction-compression."""
    # Problem size
    n_sites = 10
    phys_dim = 4
    chi_out = 40
    array_type = np.complex128

    # MPOs
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Benchmark the application
    result_mpo = benchmark(apply, H1, H2, chi_out=chi_out, dtype=array_type)

    # Still has to be correct
    np.testing.assert_allclose(H1.distance(result_mpo), 0.0, atol=1e-6)
