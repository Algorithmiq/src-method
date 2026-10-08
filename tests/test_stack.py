"""Test `src` over stacks of tensor trains, and the sweep behind it."""

import threading
from functools import reduce

import numpy as np
import pytest

import src_method._sweep as sweep_module
from src_method import Resources, apply, compress, src
from src_method._kernels import SiteKernels
from src_method._plan import make_plan
from src_method._sweep import sweep

# -------------
# --- Utils ---
# -------------


def random_mpo(bonds, rng, *, up=2, down=2):
    """Complex MPO with the given (possibly jagged) bonds, ``len(bonds) + 1`` sites."""

    def site(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    lefts, rights = [None, *bonds], [*bonds, None]
    return [
        site(*(b for b in (lb, rb) if b is not None), up, down)
        for lb, rb in zip(lefts, rights)
    ]


def random_mps(bonds, rng, *, phys=2):
    """Complex MPS with the given (possibly jagged) bonds."""
    return [t[..., 0] for t in random_mpo(bonds, rng, up=phys, down=1)]


def identity_mpo(n_sites, phys=2):
    eye = np.eye(phys, dtype=complex)
    return [eye[None]] + [eye[None, None]] * (n_sites - 2) + [eye[None]]


def dense(train):
    """Contract a train into a ``(U, D)`` matrix; an MPS gives a column ``(U, 1)``."""
    if train[0].ndim == 2:
        train = [t[..., None] for t in train]
    T = train[0]
    for W in train[1:-1]:
        T = np.einsum("aUD,abud->bUuDd", T, W)
        T = T.reshape(W.shape[1], T.shape[1] * T.shape[2], T.shape[3] * T.shape[4])
    T = np.einsum("aUD,aud->UuDd", T, train[-1])
    return T.reshape(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3])


def dense_stack(*trains):
    """Dense product of a stack; a leading MPS is a row vector, not conjugated."""
    mats = [dense(t) for t in trains]
    if len(trains) > 1 and trains[0][0].ndim == 2:
        mats[0] = mats[0].T
    return reduce(np.matmul, mats)


def rel_error(train, reference):
    got = dense(train).ravel()
    return np.linalg.norm(got - reference.ravel()) / np.linalg.norm(reference)


N_SITES = 5
CHI_EXACT = 64  # above every exact bond of the stacks below


def make_stack(spec, rng, n_sites=N_SITES):
    """Build a stack from a spec such as ``"A B psi"`` or ``"phi A B"``.

    Names ``psi`` and ``phi`` are MPSs, anything else an MPO; bonds are jagged.
    """
    mps_bonds, mpo_bonds = [3, 2, 4, 3][: n_sites - 1], [2, 3, 1, 2][: n_sites - 1]
    return [
        random_mps(mps_bonds, rng)
        if name in {"psi", "phi"}
        else random_mpo(mpo_bonds, rng)
        for name in spec.split()
    ]


KET_STACKS = ["A", "psi", "A psi", "A B", "A B psi", "A B C", "A B C psi"]
BRA_STACKS = ["phi A", "phi A B"]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ----------------------
# --- Generic sweep ---
# ----------------------


@pytest.mark.parametrize("spec", KET_STACKS)
def test_sweep_is_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)
    kind = "mps" if stack[-1][0].ndim == 2 else "mpo"

    out = sweep(stack, kind, CHI_EXACT, np.random.default_rng(0), np, dtype=np.float64)

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_sweep_rectangular_physical_legs(rng):
    A = random_mpo([2, 3, 2, 2], rng, up=2, down=3)
    B = random_mpo([3, 2, 2, 1], rng, up=3, down=2)
    psi = random_mps([2, 2, 3, 2], rng, phys=2)

    out = sweep(
        [A, B, psi], "mps", CHI_EXACT, np.random.default_rng(0), np, dtype=np.float64
    )

    assert rel_error(out, dense_stack(A, B, psi)) < 1e-10


def test_sweep_does_not_mutate_inputs(rng):
    stack = make_stack("A B psi", rng)
    before = [[t.copy() for t in train] for train in stack]

    sweep(stack, "mps", 4, np.random.default_rng(0), np, dtype=np.float64)

    for train, saved in zip(stack, before):
        for t, s in zip(train, saved):
            np.testing.assert_array_equal(t, s)


# --------------------
# --- Public src ----
# --------------------


@pytest.mark.parametrize("spec", KET_STACKS + BRA_STACKS)
def test_src_is_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)

    out = src(*stack, chi_out=CHI_EXACT, dtype=np.complex128)

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_src_bra_is_not_conjugated(rng):
    phi, A = make_stack("phi A", rng)

    out = dense(src(phi, A, chi_out=CHI_EXACT)).ravel()

    plain = (dense(phi).T @ dense(A)).ravel()
    conjugated = (dense(phi).conj().T @ dense(A)).ravel()
    np.testing.assert_allclose(out, plain, atol=1e-10)
    assert np.linalg.norm(out - conjugated) > 1e-2 * np.linalg.norm(plain)


def test_src_identity_layers(rng):
    A = random_mpo([2, 3, 3, 2], rng)
    eye = identity_mpo(N_SITES)

    out = src(eye, eye, A, chi_out=3)

    assert rel_error(out, dense(A)) < 1e-10


def test_src_truncates_to_chi_out(rng):
    stack = make_stack("A B C", rng)

    out = src(*stack, chi_out=5, seed=0)

    assert max(t.shape[0] for t in out[1:]) <= 5


def cut_lower_bound(reference, n_sites, chi):
    """Largest best-rank-``chi`` tail over all cuts of an MPS-shaped reference.

    No train of bond dimension ``chi`` can have a smaller relative error.
    """
    vec = reference.ravel()
    tails = (
        np.linalg.svd(vec.reshape(2**cut, -1), compute_uv=False)[chi:]
        for cut in range(1, n_sites)
    )
    return max(np.linalg.norm(tail) for tail in tails) / np.linalg.norm(vec)


def test_src_truncation_is_near_optimal():
    """Under truncation the error stays within a small factor of the optimum.

    Single instances have a heavy tail (up to ~10x the bound over 200 seeds), so
    the assertion is on the median over fixed instances, which sits near 2x.
    """
    chi = 2
    ratios = []
    for seed in range(10):
        stack = make_stack("A B psi", np.random.default_rng(seed))
        reference = dense_stack(*stack)
        error = rel_error(src(*stack, chi_out=chi, seed=seed), reference)
        bound = cut_lower_bound(reference, N_SITES, chi)
        assert error >= bound * (1 - 1e-10)
        ratios.append(error / bound)

    assert np.median(ratios) <= 4


@pytest.mark.parametrize("spec", ["A B C", "A B psi", "phi A B"])
def test_src_output_is_right_canonical(spec, rng):
    out = src(*make_stack(spec, rng), chi_out=3, seed=0)

    for site in out[1:]:
        flat = site.reshape(site.shape[0], -1)
        np.testing.assert_allclose(
            flat @ flat.conj().T, np.eye(site.shape[0]), atol=1e-10
        )


def test_src_does_not_mutate_bra_stack(rng):
    stack = make_stack("phi A B", rng)
    before = [[t.copy() for t in train] for train in stack]

    src(*stack, chi_out=3, seed=0)

    for train, saved in zip(stack, before):
        for t, s in zip(train, saved):
            np.testing.assert_array_equal(t, s)


def test_src_cutoff_depth_three(rng):
    eye = identity_mpo(N_SITES)
    psi = random_mps([2, 4, 4, 2], rng)

    out = src(eye, eye, psi, chi_out=16, cutoff=1e-10, seed=0)
    full = src(eye, eye, psi, chi_out=16, seed=0)

    assert rel_error(out, dense(psi)) < 1e-8
    assert all(o.shape[0] <= f.shape[0] for o, f in zip(out[1:], full[1:]))
    assert max(t.shape[0] for t in out[1:]) <= 4


def test_src_seed_is_deterministic(rng):
    stack = make_stack("A B psi", rng)

    first = src(*stack, chi_out=4, seed=7)
    second = src(*stack, chi_out=4, seed=7)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("spec", ["A B psi", "A B C", "phi A B"])
def test_src_two_site_stack_falls_back(spec, rng, caplog):
    stack = make_stack(spec, rng, n_sites=2)

    out = src(*stack, chi_out=CHI_EXACT)

    assert "Defaulting" in caplog.text
    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_src_single_site_raises_before_warning(caplog):
    mpo = [np.ones((1, 2, 2))]
    mps = [np.ones((1, 2))]

    with pytest.raises(ValueError, match="two-site tensor train"):
        src(mpo, mpo, mps, chi_out=4)

    assert "Defaulting" not in caplog.text


def test_src_validates_the_stack(rng):
    with pytest.raises(TypeError, match="MPS may come first"):
        src(*make_stack("psi A psi", rng), chi_out=4)


# ----------------
# --- Wrappers ---
# ----------------


def assert_same_trains(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def assert_cutoff_trims(train, untrimmed):
    """Guard for the wrapper tests: the cutoff must change the result."""
    assert [t.shape for t in train] != [t.shape for t in untrimmed]


@pytest.mark.parametrize("cutoff", [0.0, 0.5])
@pytest.mark.parametrize("spec", ["A psi", "A B"])
def test_apply_matches_src(spec, cutoff, rng):
    left, right = make_stack(spec, rng)

    via_apply = apply(left, right, 4, cutoff=cutoff, dtype=np.complex128, seed=3)
    via_src = src(left, right, chi_out=4, cutoff=cutoff, dtype=np.complex128, seed=3)

    assert_same_trains(via_apply, via_src)
    if cutoff:
        assert_cutoff_trims(via_src, src(left, right, chi_out=4, seed=3))


@pytest.mark.parametrize("cutoff", [0.0, 0.5])
@pytest.mark.parametrize("spec", ["psi", "A"])
def test_compress_matches_src(spec, cutoff, rng):
    (train,) = make_stack(spec, rng)

    via_compress = compress(train, 4, cutoff=cutoff, seed=3)
    via_src = src(train, chi_out=4, cutoff=cutoff, seed=3)

    assert_same_trains(via_compress, via_src)
    if cutoff:
        assert_cutoff_trims(via_src, src(train, chi_out=4, seed=3))


def test_apply_rejects_bra(rng):
    phi, A = make_stack("phi A", rng)

    with pytest.raises(TypeError, match="expected an MPO on the left"):
        apply(phi, A, 4)


# -----------------------------------------
# --- Check for performance regressions ---
# -----------------------------------------


@pytest.mark.perf
def test_benchmark_src_stack_depth3(benchmark):
    """Benchmarks a depth-3 stack: two near-identity MPOs applied to an MPS."""
    qtn = pytest.importorskip("quimb.tensor")
    n_sites, phys_dim, chi_out = 10, 2, 32
    dtype = np.complex128
    layers = [
        qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=dtype)
        + 1e-8
        * qtn.MPO_rand(n_sites, bond_dim=3, phys_dim=phys_dim, dtype=dtype, seed=seed)
        for seed in (1, 2)
    ]
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=dtype, seed=3
    )

    out = benchmark(
        src, *(t.arrays for t in layers), psi.arrays, chi_out=chi_out, dtype=dtype
    )

    # Still has to be correct
    ref = layers[0].apply(layers[1].apply(psi, compress=False), compress=False)
    np.testing.assert_allclose(
        ref.distance(qtn.MatrixProductState(out)), 0.0, atol=1e-6
    )


# ----------------------------
# --- Budgets and spilling ---
# ----------------------------


@pytest.fixture
def plans(monkeypatch):
    """Record the plan of every sweep."""
    recorded = []

    def spy(*args, **kwargs):
        recorded.append(make_plan(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(sweep_module, "make_plan", spy)
    return recorded


def test_tiny_budget_batches_and_spills(rng, tmp_path, plans):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]
    tight = Resources(host_memory="1MB", scratch_dir=tmp_path)

    default = src(*stack, chi_out=16, seed=3, dtype=np.complex128)
    spilled = src(*stack, chi_out=16, seed=3, dtype=np.complex128, resources=tight)

    assert all(site.tier == "device" for site in plans[0].sites)
    assert "disk" in {site.tier for site in plans[1].sites}
    assert min(site.sketch_batch for site in plans[1].sites[1:]) < 16
    # Rounding differs, and the output cores of an ill-conditioned sketch with it,
    # but not the operator they represent.
    assert rel_error(spilled, dense(default)) < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_budget_too_small_raises(rng):
    stack = make_stack("A B psi", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        src(*stack, chi_out=8, resources=Resources(host_memory="1kB"))


def test_apply_passes_resources(rng):
    A, psi = make_stack("A psi", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        apply(A, psi, chi_out=8, resources=Resources(host_memory="1kB"))


def test_compress_passes_resources(rng):
    (A,) = make_stack("A", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        compress(A, chi_out=8, resources=Resources(host_memory="1kB"))


# -------------------------------
# --- Several (simulated) devices ---
# -------------------------------


@pytest.mark.parametrize(
    ("devices", "chi"), [(2, 16), (3, 16), (3, 17), ([2, 0, 1], 16), (4, 3)]
)
def test_devices_match_one_device(devices, chi, rng):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]

    one = src(*stack, chi_out=chi, seed=5, dtype=np.complex128)
    many = src(
        *stack,
        chi_out=chi,
        seed=5,
        dtype=np.complex128,
        resources=Resources(devices=devices),
    )

    assert [t.shape for t in many] == [t.shape for t in one]
    assert rel_error(many, dense(one)) < 1e-10


@pytest.mark.parametrize("spec", ["psi", "A psi", "A B", "phi A B"])
def test_devices_are_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)

    out = src(*stack, chi_out=CHI_EXACT, resources=Resources(devices=3))

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_devices_with_cutoff_keep_the_bonds(rng):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(3)]

    one = src(*stack, chi_out=24, cutoff=0.05, seed=2)
    many = src(*stack, chi_out=24, cutoff=0.05, seed=2, resources=Resources(devices=2))

    assert [t.shape for t in many] == [t.shape for t in one]
    assert rel_error(many, dense(one)) < 1e-10


def test_devices_spill_and_clean_up(rng, tmp_path, plans):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]
    tight = Resources(host_memory="1MB", scratch_dir=tmp_path, devices=2)

    default = src(*stack, chi_out=16, seed=3, dtype=np.complex128)
    spilled = src(*stack, chi_out=16, seed=3, dtype=np.complex128, resources=tight)

    assert "disk" in {site.tier for site in plans[1].sites}
    assert rel_error(spilled, dense(default)) < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_an_error_on_one_device_is_raised(rng, monkeypatch, tmp_path):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(2)]
    calls, lock = [0], threading.Lock()
    original = SiteKernels.sketch

    def flaky(self, *args):
        with lock:
            calls[0] += 1
            fail = calls[0] == 2
        if fail:
            msg = "sketch failed"
            raise RuntimeError(msg)
        return original(self, *args)

    monkeypatch.setattr(SiteKernels, "sketch", flaky)
    resources = Resources(host_memory="2MB", scratch_dir=tmp_path, devices=3)
    with pytest.raises(RuntimeError, match="sketch failed"):
        src(*stack, chi_out=16, seed=0, resources=resources)
    assert list(tmp_path.iterdir()) == []


def test_devices_are_validated(rng):
    stack = make_stack("A B", rng)

    with pytest.raises(ValueError, match="positive device count"):
        Resources(devices=0)
    with pytest.raises(ValueError, match="distinct"):
        Resources(devices=[1, 1])
    with pytest.raises(TypeError, match="device count or a sequence"):
        Resources(devices="0,1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="integer device ids"):
        Resources(devices=[0.0, 1.0])  # ty: ignore[invalid-argument-type]
    # Simulated devices have no upper limit.
    src(*stack, chi_out=4, resources=Resources(devices=[0, 7]))
