"""Test the lazy site source."""

from collections.abc import Sequence

import numpy as np
import pytest

from src_method import src
from src_method._sites import SiteSource, padded_shapes, site_bytes
from src_method._tensor_train import pad


class CountingTrain(Sequence):
    """A train that counts how often each site is read."""

    def __init__(self, sites):
        self.sites = sites
        self.reads = [0] * len(sites)

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, j):
        return CountingSite(self, j)


class CountingSite:
    """A lazily read site: shape and dtype at once, data on ``np.asarray``."""

    def __init__(self, train, j):
        self.train, self.j = train, j
        self.shape = train.sites[j].shape
        self.dtype = train.sites[j].dtype
        self.ndim = train.sites[j].ndim

    def __array__(self, dtype=None, copy=None):
        self.train.reads[self.j] += 1
        return np.asarray(self.train.sites[self.j], dtype=dtype)


def random_mpo(n_sites, bond, rng, phys=2):
    shapes = (
        [(bond, phys, phys)]
        + [(bond, bond, phys, phys)] * (n_sites - 2)
        + [(bond, phys, phys)]
    )
    return [rng.normal(size=s) + 1j * rng.normal(size=s) for s in shapes]


def test_padded_shapes_and_bytes_without_reading():
    rng = np.random.default_rng(0)
    train = CountingTrain(random_mpo(4, 3, rng))
    mps = [t[..., 0] for t in random_mpo(4, 2, rng)]

    shapes = padded_shapes([train, mps])

    assert shapes == [
        tuple(t.shape for t in sites) for sites in zip(pad(train.sites), pad(mps))
    ]
    assert site_bytes([train]) == [t.nbytes for t in train.sites]
    assert train.reads == [0, 0, 0, 0]


@pytest.mark.parametrize("depth", [0, 1])
def test_sites_are_read_when_requested(depth):
    rng = np.random.default_rng(1)
    train = CountingTrain(random_mpo(4, 3, rng))

    with SiteSource([train], np, depth=depth) as source:
        (core,) = source[2]
        assert train.reads == [0, 0, 1, 0]
        source.prefetch(3)
        (last,) = source[3]
        assert train.reads == [0, 0, 1, 1]

    np.testing.assert_array_equal(core, pad(train.sites)[2])
    np.testing.assert_array_equal(last, pad(train.sites)[3])


def test_src_reads_each_site_once_per_pass():
    rng = np.random.default_rng(2)
    train = CountingTrain(random_mpo(5, 3, rng))
    other = random_mpo(5, 2, rng)

    lazy = src(CountingTrain(other), train, chi_out=4, seed=0)
    eager = src(other, train.sites, chi_out=4, seed=0)

    # Left-to-right reads sites 0..3, right-to-left sites 4..0.
    assert train.reads == [2, 2, 2, 2, 1]
    for a, b in zip(lazy, eager):
        np.testing.assert_array_equal(a, b)


def test_src_on_memmaps(tmp_path):
    rng = np.random.default_rng(3)
    trains = [random_mpo(5, 3, rng), random_mpo(5, 2, rng)]
    mapped = []
    for t, train in enumerate(trains):
        sites = []
        for j, site in enumerate(train):
            path = tmp_path / f"train{t}-site{j}.npy"
            np.save(path, site)
            sites.append(np.load(path, mmap_mode="r"))
        mapped.append(sites)

    lazy = src(*mapped, chi_out=4, seed=0)
    eager = src(*trains, chi_out=4, seed=0)

    for a, b in zip(lazy, eager):
        np.testing.assert_array_equal(a, b)


def test_src_bra_stack_of_lazy_sites():
    rng = np.random.default_rng(4)
    phi = [t[..., 0] for t in random_mpo(5, 2, rng)]
    mpo = random_mpo(5, 3, rng)

    lazy = src(phi, CountingTrain(mpo), chi_out=4, seed=0)
    eager = src(phi, mpo, chi_out=4, seed=0)

    for a, b in zip(lazy, eager):
        np.testing.assert_array_equal(a, b)
