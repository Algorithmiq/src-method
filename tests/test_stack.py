"""Test `src` over stacks of tensor trains, and the sweep behind it."""

from functools import reduce

import numpy as np
import pytest

from src_method import apply, compress, src
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

    out = sweep(stack, kind, CHI_EXACT, np.random.default_rng(0), np)

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_sweep_rectangular_physical_legs(rng):
    A = random_mpo([2, 3, 2, 2], rng, up=2, down=3)
    B = random_mpo([3, 2, 2, 1], rng, up=3, down=2)
    psi = random_mps([2, 2, 3, 2], rng, phys=2)

    out = sweep([A, B, psi], "mps", CHI_EXACT, np.random.default_rng(0), np)

    assert rel_error(out, dense_stack(A, B, psi)) < 1e-10


def test_sweep_does_not_mutate_inputs(rng):
    stack = make_stack("A B psi", rng)
    before = [[t.copy() for t in train] for train in stack]

    sweep(stack, "mps", 4, np.random.default_rng(0), np)

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


@pytest.mark.parametrize("spec", ["A psi", "A B"])
def test_apply_matches_src(spec, rng):
    left, right = make_stack(spec, rng)

    via_apply = apply(left, right, 4, dtype=np.complex128, seed=3)
    via_src = src(left, right, chi_out=4, dtype=np.complex128, seed=3)

    for a, b in zip(via_apply, via_src):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("spec", ["psi", "A"])
def test_compress_matches_src(spec, rng):
    (train,) = make_stack(spec, rng)

    via_compress = compress(train, 4, seed=3)
    via_src = src(train, chi_out=4, seed=3)

    for a, b in zip(via_compress, via_src):
        np.testing.assert_array_equal(a, b)


def test_apply_rejects_bra(rng):
    phi, A = make_stack("phi A", rng)

    with pytest.raises(TypeError, match="expected an MPO on the left"):
        apply(phi, A, 4)
