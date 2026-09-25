"""Test the lazy site source."""

import numpy as np
import pytest

from src_method._sites import SiteSource, padded_shapes, site_bytes
from src_method._tensor_train import pad


class CountingTrain:
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
