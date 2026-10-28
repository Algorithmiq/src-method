"""Chebyshev MPS spectral functions with src-method against quimb, on one GPU.

Dynamical structure factor ``S(q, w)`` of spin-1/2 Heisenberg models by the
Chebyshev matrix-product-state method (Holzner et al., arXiv:1101.5895). Every
recursion step applies the MPO to an MPS and compresses a sum of three MPS:

    t_1 = H' t_0,   t_{n+1} = 2 H' t_n - t_{n-1},   H' = (H - E_0) / a - W',

with moments from the doubling trick, ``mu_2n = 2 <t_n|t_n> - mu_0`` and
``mu_2n+1 = 2 <t_n+1|t_n> - mu_1``.

Commands:

- ``check``: `apply` against dense ``H|t>`` on 8 sites, the direct sum and the
  quimb pipelines, and the TeNPy cylinder MPO against a dense Hamiltonian;
- ``prep``: DMRG ground state and top energy, and the probe states ``S^z_q|psi_0>``,
  saved to scratch;
- ``exact``: exact Chebyshev moments by sparse matrix-vector products (small L);
- ``run``: Chebyshev recursions for a list of configurations, resumable and
  stopped before a time budget runs out, one JSON line per step;
- ``report``: tables and plots.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cyclopts
import numpy as np

import src_method
from src_method import Resources, apply, compress

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Chebyshev MPS spectral functions: src-method against quimb.")

HERE = Path(__file__).parent
OUT = HERE / "leonardo"
SCRATCH = Path(os.environ.get("CINECA_SCRATCH", "/tmp")) / "chebyshev"
W_PRIME = 0.9875
METHODS = (
    "src-method",
    "src-method-os",
    "src-method+fit1",
    "src-method-g4",
    "quimb-zipup",
    "quimb-zipup-first",
    "quimb-src",
    "quimb-fit",
)

SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
I2 = np.eye(2, dtype=complex)

Train = list[np.ndarray]


# ---------------------------------------------------------------------------
# Trains in the quimb layout: MPO (l, r, u, d), MPS (l, r, u), boundary sites
# without the outer bond. "Full" form pads the boundaries with unit bonds.
# ---------------------------------------------------------------------------


def full(train: Train) -> Train:
    """Pad the boundary sites with unit outer bonds."""
    return [train[0][None], *train[1:-1], train[-1][:, None]]


def trim(train: Train) -> Train:
    """Drop the unit outer bonds of a full-form train."""
    return [train[0][0], *train[1:-1], train[-1][:, 0]]


def bond(train: Sequence[Any]) -> int:
    """Largest bond dimension of a quimb-layout train."""
    return max(t.shape[0] for t in train[1:])


def direct_sum(terms: Sequence[tuple[complex, Train]]) -> Train:
    """``sum_k c_k x_k`` of MPS as one MPS of block-diagonal sites.

    The coefficients go on the first site; bonds add up.
    """
    fulls = [full(x) for _, x in terms]
    n = len(fulls[0])
    out = []
    for j in range(n):
        blocks = [f[j] for f in fulls]
        if j == 0:
            site = np.concatenate(
                [c * b for (c, _), b in zip(terms, blocks, strict=True)], axis=1
            )
        elif j == n - 1:
            site = np.concatenate(blocks, axis=0)
        else:
            ls, rs = [b.shape[0] for b in blocks], [b.shape[1] for b in blocks]
            site = np.zeros((sum(ls), sum(rs), blocks[0].shape[2]), dtype=complex)
            lo = ro = 0
            for b in blocks:
                site[lo : lo + b.shape[0], ro : ro + b.shape[1]] = b
                lo, ro = lo + b.shape[0], ro + b.shape[1]
        out.append(site)
    return trim(out)


def apply_exact(mpo: Train, mps: Train) -> Train:
    """Exact MPO-MPS product, bond ``D * chi``."""
    out = []
    for w, x in zip(full(mpo), full(mps), strict=True):
        t = np.einsum("abud,lrd->albru", w, x)
        s = t.shape
        out.append(t.reshape(s[0] * s[1], s[2] * s[3], s[4]))
    return trim(out)


def svd_compress(
    mps: Train, cutoff: float = 1e-13, max_bond: int | None = None
) -> Train:
    """Canonical SVD truncation: drop ``s < cutoff * s_max`` at every cut."""
    sites = [np.asarray(t, dtype=complex) for t in full(mps)]
    for j in range(len(sites) - 1):
        l, r, u = sites[j].shape
        Q, R = np.linalg.qr(sites[j].transpose(0, 2, 1).reshape(-1, r))
        sites[j] = Q.reshape(l, u, -1).transpose(0, 2, 1)
        sites[j + 1] = np.tensordot(R, sites[j + 1], axes=(1, 0))
    for j in range(len(sites) - 1, 0, -1):
        l, r, u = sites[j].shape
        U, S, Vh = np.linalg.svd(sites[j].reshape(l, -1), full_matrices=False)
        k = max(1, int((cutoff * S[0] < S).sum()))
        if max_bond is not None:
            k = min(k, max_bond)
        sites[j] = Vh[:k].reshape(k, r, u)
        sites[j - 1] = np.tensordot(sites[j - 1], U[:, :k] * S[:k], axes=(1, 0))
        sites[j - 1] = sites[j - 1].transpose(0, 2, 1)
    return trim(sites)


def dense_vector(mps: Train) -> np.ndarray:
    """State vector of an MPS; site 0 is the most significant."""
    v = full(mps)[0][0]  # (r, u)
    v = v.T  # (u, r)
    for x in full(mps)[1:]:
        v = np.tensordot(v, x, axes=(1, 0))  # (U, r, u)
        v = v.transpose(0, 2, 1).reshape(-1, x.shape[1])
    return v[:, 0]


def dense_operator(mpo: Train) -> np.ndarray:
    """Matrix of an MPO; rows on ``u``, site 0 most significant."""
    m = full(mpo)[0][0]  # (r, u, d)
    m = np.moveaxis(m, 0, -1)  # (u, d, r)
    for w in full(mpo)[1:]:
        t = np.einsum("abr,rsud->aubds", m, w)
        s = t.shape
        m = t.reshape(s[0] * s[1], s[2] * s[3], s[4])
    return m[..., 0]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def chain_mpo(n: int) -> Train:
    """Open spin-1/2 Heisenberg chain ``sum S_i . S_i+1`` (bond 5), quimb's MPO."""
    import quimb.tensor as qtn  # noqa: PLC0415

    H = qtn.MPO_ham_heis(n, j=1.0, S=1 / 2, cyclic=False)
    return [np.asarray(a, dtype=complex) for a in H.arrays]


def cylinder_model(lx: int, ly: int) -> Any:  # noqa: ANN401  (a TeNPy model)
    """TeNPy square-lattice Heisenberg model on an ``lx x ly`` cylinder, snake order."""
    from tenpy.models.spins import SpinModel  # noqa: PLC0415

    return SpinModel(
        {
            "lattice": "Square",
            "Lx": lx,
            "Ly": ly,
            "bc_MPS": "finite",
            "bc_y": "cylinder",
            "order": "snake",
            "S": 0.5,
            "Jx": 1.0,
            "Jy": 1.0,
            "Jz": 1.0,
            "conserve": None,
        }
    )


def cylinder_mpo(lx: int, ly: int) -> Train:
    """The TeNPy MPO in quimb's layout: ``wL, wR, p, p*`` is ``l, r, u, d``."""
    H = cylinder_model(lx, ly).H_MPO
    sites = [np.asarray(H.get_W(i).to_ndarray(), dtype=complex) for i in range(H.L)]
    sites[0] = sites[0][H.get_IdL(0)]
    sites[-1] = sites[-1][:, H.get_IdR(H.L - 1)]
    return sites


def cylinder_dense(lx: int, ly: int) -> np.ndarray:
    """Dense ``sum_<ij> S_i . S_j`` on the cylinder, in TeNPy's MPS site order."""
    lat = cylinder_model(lx, ly).lat
    n = lx * ly
    pos = {tuple(int(v) for v in lat.mps2lat_idx(i)[:2]): i for i in range(n)}
    bonds = set()
    for (x, y), i in pos.items():
        if x + 1 < lx:
            bonds.add(tuple(sorted((i, pos[x + 1, y]))))
        bonds.add(tuple(sorted((i, pos[x, (y + 1) % ly]))))

    def op(o: np.ndarray, k: int) -> np.ndarray:
        return np.kron(np.kron(np.eye(2**k), o), np.eye(2 ** (n - k - 1)))

    H = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in bonds:
        for s in (SX, SY, SZ):
            H += op(s, i) @ op(s, j)
    return H


def model_mpo(model: str, n: int) -> Train:
    """``chain`` of ``n`` sites, or ``yc4`` cylinder of ``4 x n // 4`` sites."""
    if model == "chain":
        return chain_mpo(n)
    if model == "yc4":
        return cylinder_mpo(n // 4, 4)
    msg = f"unknown model {model!r}"
    raise ValueError(msg)


def sz_q_mpo(n: int, k: int) -> Train:
    """``S^z_q = sqrt(2/(n+1)) sum_j sin(q j) S^z_j``, ``q = pi k / (n+1)``, bond 2."""
    q = np.pi * k / (n + 1)
    sites = []
    for j in range(1, n + 1):
        w = np.zeros((2, 2, 2, 2), dtype=complex)
        w[0, 0] = w[1, 1] = I2
        w[0, 1] = np.sqrt(2 / (n + 1)) * np.sin(q * j) * SZ
        sites.append(w)
    sites[0], sites[-1] = sites[0][0], sites[-1][:, 1]
    return sites


def k_index(n: int, target: str) -> int:
    """``k`` with ``q = pi k / (n+1)`` nearest to ``pi/2`` or ``pi``."""
    q = {"half": np.pi / 2, "pi": np.pi}[target]
    ks = np.arange(1, n + 1)
    return int(ks[np.argmin(abs(np.pi * ks / (n + 1) - q))])


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


@cache
def _cupy() -> Any:  # noqa: ANN401
    import cupy  # noqa: PLC0415  (GPU only)

    return cupy


@cache
def _analysis_pool() -> Any:  # noqa: ANN401
    return _cupy().cuda.MemoryPool()


@contextmanager
def analysis_pool() -> Iterator[None]:
    """Run inner products on a separate pool, so the default one sees the methods."""
    with _cupy().cuda.using_allocator(_analysis_pool().malloc):
        yield


def inner(a: Sequence[Any], b: Sequence[Any]) -> complex:
    """``<a|b>`` of two MPS (numpy or cupy sites), on the GPU."""
    cp = _cupy()
    with analysis_pool():
        fa, fb = full([cp.asarray(x) for x in a]), full([cp.asarray(x) for x in b])
        env = cp.ones((1, 1), dtype=complex)
        for x, y in zip(fa, fb, strict=True):
            tmp = cp.tensordot(env, cp.conj(x), axes=(0, 0))  # (lb, r, u)
            env = cp.tensordot(tmp, y, axes=([0, 2], [0, 2]))  # (r, s)
        return complex(env[0, 0])


def pool_gb() -> float:
    """The largest default-pool size over the visible GPUs, in GB."""
    cp = _cupy()
    sizes = []
    for gpu in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(gpu):
            sizes.append(cp.get_default_memory_pool().total_bytes())
    return max(sizes) / 1e9


def free_pools() -> None:
    """Release the cached blocks of every GPU's default pool."""
    cp = _cupy()
    for gpu in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(gpu):
            cp.get_default_memory_pool().free_all_blocks()


def sync() -> None:
    _cupy().cuda.Device().synchronize()


def provenance() -> dict[str, Any]:
    """Versions, job and GPU of this process."""
    import autoray  # noqa: PLC0415
    import quimb  # noqa: PLC0415

    cp = _cupy()
    meta = {
        "src_method": src_method.__version__,
        "quimb": quimb.__version__,
        "autoray": autoray.__version__,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "SRC_EINSUM_OPTIMIZE": os.environ.get("SRC_EINSUM_OPTIMIZE"),
    }
    try:
        import tenpy  # noqa: PLC0415

        meta["tenpy"] = tenpy.__version__
    except ImportError:
        pass
    return meta


# ---------------------------------------------------------------------------
# Pipelines: one apply and one compression of a three-term sum per step
# ---------------------------------------------------------------------------


class SrcMethod:
    """src-method: `apply` then `compress` of the explicit direct sum."""

    name = "src-method"

    def __init__(
        self,
        H: Train,
        chi: int,
        seed: int,
        *,
        oversample: float | None = 1.0,
        devices: int = 1,
    ) -> None:
        self.H, self.chi, self.seed = H, chi, seed
        self.opts: dict[str, Any] = {"device": "gpu", "oversample": oversample}
        if devices > 1:
            self.opts["resources"] = Resources(devices=devices)

    def state(self, arrays: Train) -> Train:
        return [np.asarray(a) for a in arrays]

    def apply(self, t: Train, n: int) -> Train:
        return apply(self.H, t, self.chi, seed=self._seed(2 * n), **self.opts)

    def combine(self, terms: Sequence[tuple[complex, Train]], n: int) -> Train:
        return compress(
            direct_sum(terms), self.chi, seed=self._seed(2 * n + 1), **self.opts
        )

    def arrays(self, t: Train) -> Train:
        return t

    def _seed(self, k: int) -> int:
        return self.seed * 1_000_003 + k


@cache
def _quimb_on_device() -> None:
    """Make quimb 1.15's random tensors follow the network onto the GPU.

    ``src`` draws its sketch noise, and ``fit`` its initial guess, with numpy
    whatever the arrays of the network, which then fails inside CuPy's tensordot.
    The wrappers move those random tensors to the device; the distributions,
    dtypes and shapes are quimb's own.
    """
    from quimb.tensor.tn1d import compress as qc  # noqa: PLC0415

    cp = _cupy()
    noise, matching = qc._src_get_local_noise_tensors, qc.TN_matching  # noqa: SLF001

    def noise_on_device(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        tws = noise(*args, **kwargs)
        for t in tws:
            t.modify(data=cp.asarray(t.data))
        return tws

    def matching_on_device(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        tn = matching(*args, **kwargs)
        tn.apply_to_arrays(cp.asarray)
        return tn

    qc._src_get_local_noise_tensors = noise_on_device  # noqa: SLF001
    qc.TN_matching = matching_on_device


class Quimb:
    """quimb on the GPU: lazy `tensor_network_apply_op_vec`, then the 1D compressor
    ``method`` for both the product and the explicit MPS sum (quimb's ``+``).

    Every quimb method runs with its defaults, except ``cutoff=0`` so that the
    bond is ``chi`` everywhere, as for src-method.
    """

    def __init__(self, method: str, H: Train, chi: int, seed: int) -> None:
        import quimb as qu  # noqa: PLC0415
        import quimb.tensor as qtn  # noqa: PLC0415

        _quimb_on_device()
        self.qtn, self.method, self.chi = qtn, method, chi
        self.name = f"quimb-{method}"
        cp = _cupy()
        self.H = qtn.MatrixProductOperator([cp.asarray(w) for w in H])
        qu.seed_rand(seed)
        cp.random.seed(seed)

    def state(self, arrays: Train) -> Any:  # noqa: ANN401
        cp = _cupy()
        return self.qtn.MatrixProductState([cp.asarray(a) for a in arrays])

    def _compress(self, tn: Any) -> Any:  # noqa: ANN401
        out = self.qtn.tensor_network_1d_compress(
            tn, max_bond=self.chi, cutoff=0.0, method=self.method
        )
        sync()
        return out

    def apply(self, t: Any, n: int) -> Any:  # noqa: ANN401, ARG002
        if self.method == "zipup":
            # quimb's zipup is not exact even at a bond holding the whole state
            # (diagnose). Left-canonical input helps: in the 12-site recursion at
            # full bond the error after 11 steps drops from 1e-1 to 4e-2, and at
            # L = 20 the moments converge with chi instead of diverging
            # (quimb-zipup-rightcanon runs). The QR sweep is timed.
            t = t.copy()
            t.left_canonize()
        tn = self.qtn.tensor_network_apply_op_vec(self.H, t, contract=False)
        return self._compress(tn)

    def combine(self, terms: Sequence[tuple[complex, Any]], n: int) -> Any:  # noqa: ANN401, ARG002
        total = None
        for c, x in terms:
            total = c * x if total is None else total + c * x
        return self._compress(total)

    def arrays(self, t: Any) -> list[Any]:  # noqa: ANN401
        t.permute_arrays("lrp")
        return list(t.arrays)


class SrcFit(SrcMethod):
    """src-method, then one quimb 1-site ``fit`` sweep started from its result.

    The apply is refined against the lazy ``H|t>`` network and the sum against
    its three terms directly (``fit`` takes a list of networks), so the ``3 chi``
    direct sum is never formed for the sweep. States live on the host for
    src-method and move to the GPU for the sweep; the copies are timed.
    """

    name = "src-method+fit1"

    def __init__(self, H: Train, chi: int, seed: int) -> None:
        import quimb.tensor as qtn  # noqa: PLC0415

        super().__init__(H, chi, seed)
        _quimb_on_device()
        self.qtn = qtn
        self.Hq = qtn.MatrixProductOperator([_cupy().asarray(w) for w in H])

    def _mps(self, arrays: Train) -> Any:  # noqa: ANN401
        cp = _cupy()
        return self.qtn.MatrixProductState([cp.asarray(a) for a in arrays])

    def _sweep(self, target: Any, guess: Train) -> Train:  # noqa: ANN401
        from quimb.tensor.tn1d.compress import (  # noqa: PLC0415
            tensor_network_1d_compress_fit,
        )

        out = tensor_network_1d_compress_fit(
            target,
            max_bond=self.chi,
            cutoff=0.0,
            tn_fit=self._mps(guess),
            max_iterations=1,
        )
        out.permute_arrays("lrp")
        cp = _cupy()
        return [cp.asnumpy(a) for a in out.arrays]

    def apply(self, t: Train, n: int) -> Train:
        guess = super().apply(t, n)
        lazy = self.qtn.tensor_network_apply_op_vec(
            self.Hq, self._mps(t), contract=False
        )
        return self._sweep(lazy, guess)

    def combine(self, terms: Sequence[tuple[complex, Train]], n: int) -> Train:
        guess = super().combine(terms, n)
        return self._sweep([c * self._mps(x) for c, x in terms], guess)


def pipeline(method: str, H: Train, chi: int, seed: int) -> SrcMethod | Quimb:
    if method == "src-method":
        return SrcMethod(H, chi, seed)
    if method == "src-method-os":
        return SrcMethod(H, chi, seed, oversample=None)
    if method == "src-method-g4":
        return SrcMethod(H, chi, seed, devices=4)
    if method == "src-method+fit1":
        return SrcFit(H, chi, seed)
    return Quimb(method.removeprefix("quimb-"), H, chi, seed)


# ---------------------------------------------------------------------------
# Spectral functions
# ---------------------------------------------------------------------------


def jackson(n_moments: int) -> np.ndarray:
    """Jackson kernel coefficients ``g_n``."""
    n = np.arange(n_moments)
    big = n_moments + 1
    return (
        (big - n) * np.cos(np.pi * n / big)
        + np.sin(np.pi * n / big) / np.tan(np.pi / big)
    ) / big


def spectrum(mu: np.ndarray, a: float, omega: np.ndarray) -> np.ndarray:
    """``S(w)`` from moments, ``x = w / a - W'``."""
    x = omega / a - W_PRIME
    g = jackson(len(mu)) * mu
    T = np.cos(np.outer(np.arange(len(mu)), np.arccos(np.clip(x, -1, 1))))
    s = g[0] * T[0] + 2 * (g[1:, None] * T[1:]).sum(axis=0)
    return s / (np.pi * a * np.sqrt(1 - x**2))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _scratch(tag: str) -> Path:
    path = SCRATCH / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_train(path: Path, train: Sequence[Any]) -> None:
    cp = _cupy()
    np.savez(path, *[cp.asnumpy(t) if isinstance(t, cp.ndarray) else t for t in train])


def _load_train(path: Path) -> Train:
    with np.load(path) as z:
        return [z[f"arr_{i}"] for i in range(len(z.files))]


@app.command
def check(output: Path = OUT / "checks.json") -> None:
    """Layout and correctness checks; fail loudly if any is off.

    - ``apply(H, t)`` at full ``chi`` against dense ``H|t>`` on 8 sites, 1e-12;
    - ``compress`` of the direct sum against the dense sum;
    - one step of every quimb pipeline against dense, on the GPU;
    - the TeNPy 2 x 4 cylinder MPO against the dense Hamiltonian.

    Args:
        output: JSON file for the deviations.
    """
    rng = np.random.default_rng(0)
    n = 8
    H = chain_mpo(n)
    Hd = dense_operator(H)
    shapes = [(4, 2)] + [(4, 4, 2)] * (n - 2) + [(4, 2)]
    t = [rng.normal(size=s) + 1j * rng.normal(size=s) for s in shapes]
    td = dense_vector(t)
    ref = Hd @ td
    out: dict[str, Any] = {"meta": provenance()}

    def rel(got: np.ndarray, want: np.ndarray) -> float:
        return float(np.linalg.norm(got - want) / np.linalg.norm(want))

    y = apply(H, t, 64, device="gpu", seed=0)
    out["apply_vs_dense_L8"] = rel(dense_vector(y), ref)
    out["apply_exact_vs_dense_L8"] = rel(dense_vector(apply_exact(H, t)), ref)
    s = compress(direct_sum([(2.0, y), (-0.5, t), (-1.0, t)]), 64, device="gpu", seed=1)
    out["compress_sum_vs_dense_L8"] = rel(dense_vector(s), 2 * ref - 1.5 * td)
    out["inner_vs_dense_L8"] = abs(inner(t, y) - np.vdot(td, ref)) / abs(
        np.vdot(td, ref)
    )
    for method in METHODS[1:]:
        p = pipeline(method, H, 64, 0)
        yq = p.apply(p.state(t), 0)
        got = [_cupy().asnumpy(a) for a in p.arrays(yq)]
        out[f"{method}_apply_vs_dense_L8"] = rel(dense_vector(got), ref)
        sq = p.combine([(2.0, yq), (-0.5, p.state(t)), (-1.0, p.state(t))], 0)
        got = [_cupy().asnumpy(a) for a in p.arrays(sq)]
        out[f"{method}_sum_vs_dense_L8"] = rel(dense_vector(got), 2 * ref - 1.5 * td)
    Hc = cylinder_mpo(2, 4)
    out["cylinder_2x4_D"] = bond(Hc)
    out["cylinder_2x4_vs_dense"] = float(
        np.abs(dense_operator(Hc) - cylinder_dense(2, 4)).max()
    )
    out["cylinder_4x16_D"] = bond(cylinder_mpo(16, 4))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    bad = {
        k: v
        for k, v in out.items()
        if k not in {"meta"} and not k.endswith("_D") and v > 1e-12
    }
    for k, v in out.items():
        if k != "meta":
            logger.info("check %s: %s", k, v)
    if bad:
        msg = f"checks failed: {bad}"
        raise SystemExit(msg)


@app.command
def prep(
    model: str = "chain",
    n: int = 100,
    chi_max: int = 256,
    rel_tol: float = 1e-10,
    max_sweeps: int = 40,
) -> None:
    """DMRG ground state, top energy and probe states, saved to scratch.

    Args:
        model: ``chain`` or ``yc4``.
        n: Number of sites.
        chi_max: Largest DMRG bond.
        rel_tol: Relative energy change per sweep at convergence.
        max_sweeps: DMRG sweep limit.
    """
    import quimb.tensor as qtn  # noqa: PLC0415

    tag = f"{model}_L{n}"
    folder = _scratch(tag)
    H = model_mpo(model, n)
    np.savez(folder / "H.npz", *H)
    # |E| < 0.7 n for both models; quimb's DMRG tolerance is absolute.
    tol = rel_tol * 0.7 * n
    dims = [d for d in (16, 32, 64, 128, 256, 512, 1024) if d <= chi_max]

    def dmrg(
        sign: float,
        bonds: list[int],
        p0: Any = None,  # noqa: ANN401
    ) -> tuple[float, Train, dict]:
        mpo = qtn.MatrixProductOperator([sign * H[0], *H[1:]])
        solver = qtn.DMRG2(mpo, bond_dims=bonds, cutoffs=1e-12, p0=p0)
        start = perf_counter()
        converged = solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=1)
        e = solver.energies
        psi = solver.state
        psi.permute_arrays("lrp")
        info = {
            "converged": bool(converged),
            "sweeps": len(e),
            "seconds": perf_counter() - start,
            "rel_change": float(abs(e[-1] - e[-2]) / abs(e[-1]))
            if len(e) > 1
            else None,
            "bond": int(psi.max_bond()),
        }
        return (
            sign * float(np.real(e[-1])),
            [np.asarray(a, dtype=complex) for a in psi.arrays],
            info,
        )

    ground = folder / "ground.json"
    if ground.exists() and (folder / "psi0.npz").exists():
        saved = json.loads(ground.read_text())
        e0, info0 = saved["E0"], saved["info"]
        psi0 = _load_train(folder / "psi0.npz")
        logger.info("prep %s: reusing the saved ground state", tag)
    else:
        e0, psi0, info0 = dmrg(1.0, dims)
        np.savez(folder / "psi0.npz", *psi0)
        ground.write_text(json.dumps({"E0": e0, "info": info0}))
    # The top state is the ferromagnetic multiplet. From a random start DMRG lands
    # near its Sz = 0 member, a Dicke state of bond n / 2 + 1, and on 100 sites
    # still crept at 1e-5 after 14 sweeps; |up...up> is in the multiplet.
    up = qtn.MPS_computational_state("0" * n, dtype="complex128")
    emax, _, infomax = dmrg(-1.0, [16, 32, 64, 128], p0=up)
    infomax["start"] = "|up...up>"
    a = (emax - e0) / (2 * W_PRIME)
    meta: dict[str, Any] = {
        "model": model,
        "n": n,
        "D": bond(H),
        "E0": e0,
        "Emax": emax,
        "a": a,
        "W_prime": W_PRIME,
        "dmrg_ground": info0,
        "dmrg_top": infomax,
        "provenance": provenance(),
        "probes": {},
    }
    for target in ("half", "pi"):
        k = k_index(n, target)
        t0 = svd_compress(apply_exact(sz_q_mpo(n, k), psi0), cutoff=1e-14)
        np.savez(folder / f"t0_{target}.npz", *t0)
        meta["probes"][target] = {
            "k": k,
            "q": np.pi * k / (n + 1),
            "bond": bond(t0),
            "mu0": float(np.real(inner(t0, t0))),
        }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"prep_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        "prep %s: %s",
        tag,
        json.dumps({k: v for k, v in meta.items() if k != "provenance"}),
    )


@app.command
def exact(n: int = 20, moments: int = 200) -> None:
    """Exact Chebyshev moments of the chain by sparse matrix-vector products.

    Uses the DMRG ground state and energies of ``prep``, so that only the
    recursion differs from the MPS runs.

    Args:
        n: Number of sites (2^n states).
        moments: Number of moments.
    """
    import quimb as qu  # noqa: PLC0415

    folder = SCRATCH / f"chain_L{n}"
    meta = json.loads((folder / "meta.json").read_text())
    Hs = qu.ham_heis(n, j=1.0, cyclic=False, sparse=True).astype(complex)
    psi0 = dense_vector(_load_train(folder / "psi0.npz"))
    e_dense = float(np.real(np.vdot(psi0, Hs @ psi0)) / np.real(np.vdot(psi0, psi0)))
    a, e0 = meta["a"], meta["E0"]
    out: dict[str, Any] = {"n": n, "E0_dmrg": e0, "E0_from_sparse_H": e_dense}
    for target in ("half", "pi"):
        t0 = dense_vector(_load_train(folder / f"t0_{target}.npz"))

        def hp(v: np.ndarray) -> np.ndarray:
            return (Hs @ v - e0 * v) / a - W_PRIME * v

        mu = np.zeros(moments)
        prev, cur = t0, hp(t0)
        mu[0], mu[1] = np.real(np.vdot(t0, t0)), np.real(np.vdot(t0, cur))
        k = 1
        while 2 * k < moments:
            mu[2 * k] = 2 * np.real(np.vdot(cur, cur)) - mu[0]
            nxt = 2 * hp(cur) - prev
            if 2 * k + 1 < moments:
                mu[2 * k + 1] = 2 * np.real(np.vdot(nxt, cur)) - mu[1]
            prev, cur, k = cur, nxt, k + 1
        out[target] = mu.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exact_chain_L{n}.json").write_text(json.dumps(out) + "\n")
    logger.info("exact moments: E0 dmrg %.12f, sparse %.12f", e0, e_dense)


@app.command
def diagnose(output: Path = OUT / "diagnose_zipup.json") -> None:
    """Why quimb's ``zipup`` fails here: exactness at full bond, on the CPU.

    On 12 sites ``chi = 64`` holds any state exactly, so every apply and every
    recursion step should be exact to rounding.

    Args:
        output: JSON file for the errors.
    """
    import quimb.tensor as qtn  # noqa: PLC0415

    n, chi = 12, 64
    H = chain_mpo(n)
    Hq, Hd = qtn.MatrixProductOperator(H), dense_operator(H)
    methods = ("zipup", "zipup-first", "src", "fit")
    out: dict[str, Any] = {"n": n, "chi": chi, "apply_by_gauge": {}, "recursion": {}}
    x = qtn.MPS_rand_state(n, 64, dtype="complex128", seed=7)
    ref = Hd @ x.to_dense().ravel()
    for form in ("as-built", "right-canonical", "left-canonical"):
        xx = x.copy()
        if form == "right-canonical":
            xx.right_canonize()
        elif form == "left-canonical":
            xx.left_canonize()
        out["apply_by_gauge"][form] = {}
        for m in methods:
            tn = qtn.tensor_network_apply_op_vec(Hq, xx, contract=False)
            o = qtn.tensor_network_1d_compress(tn, max_bond=chi, cutoff=0.0, method=m)
            err = np.linalg.norm(o.to_dense().ravel() - ref) / np.linalg.norm(ref)
            out["apply_by_gauge"][form][m] = float(err)
    x = qtn.MPS_rand_state(n, 8, dtype="complex128", seed=3)
    a, e0 = 6.0, -5.0

    def hp(v: np.ndarray) -> np.ndarray:
        return (Hd @ v - e0 * v) / a - W_PRIME * v

    for m in methods:

        def comp(tn: Any, m: str = m) -> Any:  # noqa: ANN401
            return qtn.tensor_network_1d_compress(
                tn, max_bond=chi, cutoff=0.0, method=m
            )

        y = comp(qtn.tensor_network_apply_op_vec(Hq, x, contract=False))
        cur, prev = comp((1 / a) * y + (-(e0 / a + W_PRIME)) * x), x
        pe, ce = x.to_dense().ravel(), hp(x.to_dense().ravel())
        errs = []
        for _ in range(1, 12):
            y = comp(qtn.tensor_network_apply_op_vec(Hq, cur, contract=False))
            nxt = comp((2 / a) * y + (-2 * (e0 / a + W_PRIME)) * cur + (-1.0) * prev)
            pe, ce = ce, 2 * hp(ce) - pe
            prev, cur = cur, nxt
            errs.append(
                float(np.linalg.norm(cur.to_dense().ravel() - ce) / np.linalg.norm(ce))
            )
        out["recursion"][m] = errs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("diagnose: %s", json.dumps(out))


def _warmup(methods: Sequence[str]) -> None:
    """Compile the CuPy kernels of every method before timing."""
    rng = np.random.default_rng(1)
    H = chain_mpo(8)
    shapes = [(4, 2)] + [(4, 4, 2)] * 6 + [(4, 2)]
    t = [rng.normal(size=s) + 0j for s in shapes]
    for method in methods:
        p = pipeline(method, H, 8, 0)
        x = p.state(t)
        y = p.apply(x, 0)
        p.combine([(1.0, y), (1.0, x), (-1.0, x)], 0)


@app.command
def run(
    configs: tuple[str, ...],
    *,
    steps: int = 100,
    budget_minutes: float = 25.0,
    checkpoint_every: int = 10,
    output_dir: Path = OUT / "runs",
) -> None:
    """Chebyshev recursions, one per configuration, in order, within a time budget.

    A configuration is ``model:n:target:method:chi:seed``, e.g.
    ``chain:100:pi:src-method:512:0``. Each appends one JSON line per step to its
    file and checkpoints ``t_n, t_n-1`` to scratch; a rerun resumes it, and a
    finished one is skipped. Before a step could overrun the budget the command
    checkpoints and stops.

    Args:
        configs: The configurations.
        steps: Recursion steps ``n``; moments ``mu_0 .. mu_2n``.
        budget_minutes: Wall-time budget of this invocation.
        checkpoint_every: Steps between checkpoints.
        output_dir: Folder of the JSON lines.
    """
    start_job = perf_counter()
    cp = _cupy()
    _warmup(sorted({c.split(":")[3] for c in configs}))
    meta = provenance()
    output_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        model, n_s, target, method, chi_s, seed_s = config.split(":")
        n, chi, seed = int(n_s), int(chi_s), int(seed_s)
        tag = f"{model}_L{n}"
        folder = SCRATCH / tag
        pmeta = json.loads((folder / "meta.json").read_text())
        a, e0 = pmeta["a"], pmeta["E0"]
        shift = e0 / a + W_PRIME
        name = f"{tag}_{target}_{method}_chi{chi}_s{seed}"
        path = output_dir / f"{name}.jsonl"
        ckpt = SCRATCH / "ckpt" / name
        ckpt.mkdir(parents=True, exist_ok=True)
        rows = (
            [json.loads(x) for x in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        rows = [r for r in rows if "meta" not in r]
        done = (
            json.loads((ckpt / "state.json").read_text())
            if (ckpt / "state.json").exists()
            else None
        )
        if done and done["n"] >= steps:
            logger.info("%s: complete", name)
            continue
        try:
            # Drop the previous configuration's states before resetting the pool.
            p = t0 = prev = cur = nxt = y = None
            free_pools()
            H = _load_train(folder / "H.npz")
            p = pipeline(method, H, chi, seed)
            t0 = p.state(_load_train(folder / f"t0_{target}.npz"))
            if done:
                rows = [r for r in rows if r["n"] <= done["n"]]
                prev = p.state(_load_train(ckpt / "prev.npz"))
                cur = p.state(_load_train(ckpt / "cur.npz"))
                mu0, mu1, n0 = done["mu0"], done["mu1"], done["n"]
                logger.info("%s: resuming after step %d", name, n0)
            else:
                mu0 = float(np.real(inner(p.arrays(t0), p.arrays(t0))))
                t_start = perf_counter()
                y = p.apply(t0, 0)
                t_apply = perf_counter() - t_start
                t_start = perf_counter()
                cur = p.combine([(1 / a, y), (-shift, t0)], 0)
                t_comb = perf_counter() - t_start
                del y
                mu1 = float(np.real(inner(p.arrays(cur), p.arrays(t0))))
                prev, n0 = t0, 1
                rows = [
                    {
                        "n": 1,
                        "apply_s": t_apply,
                        "compress_s": t_comb,
                        "moments": {"0": mu0, "1": mu1},
                        "bond": bond(p.arrays(cur)),
                        "pool_gb": pool_gb(),
                    }
                ]
            with path.open("w") as f:
                f.write(
                    json.dumps({"meta": meta | {"config": config, "a": a, "E0": e0}})
                    + "\n"
                )
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                slowest = max(
                    (r["apply_s"] + r["compress_s"] for r in rows), default=0.0
                )
                stopped = False
                for k in range(n0, steps):
                    elapsed = perf_counter() - start_job
                    if elapsed + 1.5 * slowest + 60 > 60 * budget_minutes:
                        stopped = True
                        break
                    # cur = t_k, prev = t_k-1; make t_k+1.
                    t_start = perf_counter()
                    y = p.apply(cur, k)
                    t_apply = perf_counter() - t_start
                    t_start = perf_counter()
                    nxt = p.combine([(2 / a, y), (-2 * shift, cur), (-1.0, prev)], k)
                    t_comb = perf_counter() - t_start
                    del y
                    ca = p.arrays(cur)
                    mus = {
                        str(2 * k): 2 * float(np.real(inner(ca, ca))) - mu0,
                        str(2 * k + 1): 2 * float(np.real(inner(p.arrays(nxt), ca)))
                        - mu1,
                    }
                    row = {
                        "n": k + 1,
                        "apply_s": t_apply,
                        "compress_s": t_comb,
                        "moments": mus,
                        "bond": bond(p.arrays(nxt)),
                        "pool_gb": pool_gb(),
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows.append(row)
                    slowest = max(slowest, t_apply + t_comb)
                    prev, cur = cur, nxt
                    if (k + 1) % checkpoint_every == 0 or k + 1 == steps:
                        _checkpoint(ckpt, p, prev, cur, k + 1, mu0, mu1)
                else:
                    k = steps
                if stopped:
                    _checkpoint(ckpt, p, prev, cur, k, mu0, mu1)
                    logger.info(
                        "%s: budget reached at step %d; rerun to resume", name, k
                    )
                    return
            logger.info(
                "%s: done, median apply %.3f s, compress %.3f s",
                name,
                median(r["apply_s"] for r in rows[1:] or rows),
                median(r["compress_s"] for r in rows[1:] or rows),
            )
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as err:
            # Out of memory is a result here (capacity probes), not a failure.
            last = rows[-1]["n"] if rows else 0
            with path.open("a") as f:
                f.write(
                    json.dumps({"oom": True, "n": last + 1, "error": str(err)[:300]})
                    + "\n"
                )
            logger.info("%s: out of memory at step %d", name, last + 1)
            p = t0 = prev = cur = nxt = None
            free_pools()


def _checkpoint(
    ckpt: Path, p: Any, prev: Any, cur: Any, n: int, mu0: float, mu1: float
) -> None:  # noqa: ANN401
    _save_train(ckpt / "prev.npz", p.arrays(prev))
    _save_train(ckpt / "cur.npz", p.arrays(cur))
    (ckpt / "state.json").write_text(json.dumps({"n": n, "mu0": mu0, "mu1": mu1}))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Categorical slots 1-4 of the reference palette, fixed per method.
COLORS = {
    "src-method": "#2a78d6",
    "src-method-os": "#6250d6",
    "src-method+fit1": "#008300",
    "src-method-g4": "#e34948",
    "quimb-zipup": "#eb6834",
    "quimb-zipup-first": "#e87ba4",
    "quimb-src": "#1baf7a",
    "quimb-fit": "#eda100",
}
MARKERS = {
    "src-method": "o",
    "src-method-os": "P",
    "src-method+fit1": "X",
    "src-method-g4": "h",
    "quimb-zipup": "s",
    "quimb-zipup-first": "v",
    "quimb-src": "^",
    "quimb-fit": "D",
}
OMEGA = np.linspace(0.0, 4.0, 801)


def _runs(folder: Path) -> list[dict[str, Any]]:
    """Every run: its configuration, per-step rows and moments."""
    runs = []
    for path in sorted(folder.glob("*.jsonl")):
        lines = []
        for x in path.read_text().splitlines():
            try:
                lines.append(json.loads(x))
            except json.JSONDecodeError:  # a line still being written
                continue
        head = next((r["meta"] for r in lines if "meta" in r), None)
        rows = [r for r in lines if "meta" not in r and "oom" not in r]
        oom = next((r["n"] for r in lines if r.get("oom")), None)
        if head is None or not rows:
            continue
        model, n, target, method, chi, seed = head["config"].split(":")
        moments: dict[int, float] = {}
        for r in rows:
            moments |= {int(k): v for k, v in r["moments"].items()}
        count = 0
        while count in moments:
            count += 1
        runs.append(
            {
                "model": model,
                "n": int(n),
                "target": target,
                "method": method,
                "chi": int(chi),
                "seed": int(seed),
                "rows": rows,
                "mu": np.array([moments[i] for i in range(count)]),
                "a": head["a"],
                "meta": head,
                "oom": oom,
            }
        )
    return runs


def _timing(run: dict[str, Any]) -> dict[str, float]:
    """Medians over steps 2.. (the first is a single-term sum)."""
    rows = run["rows"][1:] or run["rows"]
    apply_s = median(r["apply_s"] for r in rows)
    comp_s = median(r["compress_s"] for r in rows)
    return {
        "apply_s": apply_s,
        "compress_s": comp_s,
        # Each step yields two moments.
        "per_moment_s": (apply_s + comp_s) / 2,
        "total_s": sum(r["apply_s"] + r["compress_s"] for r in run["rows"]),
        "pool_gb": max(r["pool_gb"] for r in run["rows"]),
        "steps": len(run["rows"]),
    }


def _edges(q: float) -> tuple[float, float]:
    """des Cloizeaux-Pearson lower edge and two-spinon upper edge, ``J = 1``."""
    return np.pi / 2 * abs(np.sin(q)), np.pi * abs(np.sin(q / 2))


def _plot_setup() -> Any:  # noqa: ANN401
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _style(ax: Any) -> None:  # noqa: ANN401
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@app.command
def report(
    folder: Path = OUT,
    reference: str = "src-method",
    n_moments: int = 0,
) -> None:
    """Write ``summary.md`` and plots from the checks, preps and runs.

    The reference of each (model, target) is the largest ``chi`` of
    ``reference`` (seed 0), checked against the next ``chi`` down. A hand-written
    ``conclusion.md`` in ``folder``, if present, is appended.

    Args:
        folder: The output folder.
        reference: Method whose largest ``chi`` is the reference.
        n_moments: Moments compared (0: the most every run of a group has).
    """
    plt = _plot_setup()
    runs = _runs(folder / "runs")
    lines = ["# Chebyshev MPS spectral functions: src-method against quimb", ""]

    metas = [r["meta"] for r in runs]
    if metas:
        m = metas[0]
        lines += [
            "## Provenance",
            "",
            f"src-method {m['src_method']}, quimb {m['quimb']}, CuPy {m['cupy']}, "
            f"NumPy {m['numpy']}, CUDA runtime {m['cuda_runtime']}, driver "
            f"{m['cuda_driver']}, {m['gpu']}"
            + (f", TeNPy {m['tenpy']}" if "tenpy" in m else "")
            + ". `SRC_EINSUM_OPTIMIZE` unset (src-method's default `dp` paths). "
            f"Jobs: {', '.join(sorted({str(x['slurm_job']) for x in metas}))}.",
            "",
        ]
    checks = folder / "checks.json"
    if checks.exists():
        c = json.loads(checks.read_text())
        lines += ["## Checks", "", "| check | value |", "|---|---|"]
        lines += [
            f"| {k} | {v:.2e} |" if isinstance(v, float) else f"| {k} | {v} |"
            for k, v in c.items()
            if k != "meta"
        ]
        lines.append("")
    preps = sorted(folder.glob("prep_*.json"))
    if preps:
        lines += [
            "## Ground states",
            "",
            "| model | sites | D | E0 | Emax | a | DMRG sweeps | rel. change | bond"
            " | q (k) | t0 bond | mu0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for path in preps:
            pm = json.loads(path.read_text())
            g = pm["dmrg_ground"]
            for pr in pm["probes"].values():
                lines.append(
                    f"| {pm['model']} | {pm['n']} | {pm['D']} | {pm['E0']:.10f}"
                    f" | {pm['Emax']:.6f} | {pm['a']:.4f} | {g['sweeps']}"
                    f" | {g['rel_change']:.1e} | {g['bond']}"
                    f" | {pr['q'] / np.pi:.4f} pi ({pr['k']}) | {pr['bond']}"
                    f" | {pr['mu0']:.6f} |"
                )
        lines.append("")

    # Phase 1: against exact moments.
    for path in sorted(folder.glob("exact_chain_L*.json")):
        ex = json.loads(path.read_text())
        n = ex["n"]
        group = [r for r in runs if r["model"] == "chain" and r["n"] == n]
        if not group:
            continue
        lines += [
            f"## Phase 1: chain L = {n} against exact moments",
            "",
            f"DMRG E0 {ex['E0_dmrg']:.12f}; <psi0|H|psi0> with the sparse H"
            f" {ex['E0_from_sparse_H']:.12f}. Error `max_n |mu_n - mu_n^exact| / mu_0`"
            " over the moments each run reached (all seeds: max).",
            "",
            "| q | method | chi | moments | max error | error at last moment |",
            "|---|---|---|---|---|---|",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, target in zip(axes, ("half", "pi"), strict=True):
            exact_mu = np.array(ex[target])
            keys = sorted(
                {(r["method"], r["chi"]) for r in group if r["target"] == target}
            )
            for method, chi in keys:
                sub = [
                    r
                    for r in group
                    if r["target"] == target
                    and r["method"] == method
                    and r["chi"] == chi
                ]
                errs = []
                for r in sub:
                    k = min(len(r["mu"]), len(exact_mu))
                    errs.append(abs(r["mu"][:k] - exact_mu[:k]) / exact_mu[0])
                k = min(len(e) for e in errs)
                worst = np.max([e[:k] for e in errs], axis=0)
                lines.append(
                    f"| {target} | {method} | {chi} | {k} | {worst.max():.2e} | {worst[-1]:.2e} |"
                )
                ax.semilogy(
                    np.arange(k),
                    np.maximum(worst, 1e-16),
                    label=f"{method} chi={chi}",
                    linewidth=1.5,
                )
            ax.set_title(
                f"L = {n}, q = {'pi/2' if target == 'half' else 'pi'}", fontsize=10
            )
            ax.set_xlabel("n")
            ax.set_ylabel("|mu_n - mu_n^exact| / mu_0 (max over seeds)")
            _style(ax)
            ax.legend(frameon=False, fontsize=7)
        name = f"phase1_L{n}_moment_error.png"
        fig.savefig(folder / name, dpi=150)
        plt.close(fig)
        lines += ["", f"![{name}]({name})", ""]

    # Phases 2 and 3: timing and accuracy against the reference.
    for model, n in sorted({(r["model"], r["n"]) for r in runs if r["n"] > 20}):
        group = [r for r in runs if r["model"] == model and r["n"] == n]
        D = None
        prep_path = folder / f"prep_{model}_L{n}.json"
        if prep_path.exists():
            D = json.loads(prep_path.read_text())["D"]
        lines += [f"## {model} L = {n} (MPO bond D = {D})", ""]
        for target in ("half", "pi"):
            tg = [r for r in group if r["target"] == target]
            if not tg:
                continue
            q = (
                np.pi
                * json.loads(prep_path.read_text())["probes"][target]["k"]
                / (n + 1)
            )
            a = tg[0]["a"]
            ref_runs = sorted(
                (r for r in tg if r["method"] == reference and r["seed"] == 0),
                key=lambda r: r["chi"],
            )
            ref = ref_runs[-1] if ref_runs else None
            # Runs with fewer moments (timing-only) show times but no errors.
            nm = n_moments or (len(ref["mu"]) if ref else min(len(r["mu"]) for r in tg))
            lines += [
                f"### q = {q / np.pi:.4f} pi, {nm} moments",
                "",
            ]
            s_ref = (
                spectrum(ref["mu"][:nm], a, OMEGA)
                if ref is not None and len(ref["mu"]) >= nm
                else None
            )
            if ref is not None and len(ref_runs) > 1 and len(ref_runs[-2]["mu"]) >= nm:
                prev = ref_runs[-2]
                d_mu = np.max(abs(prev["mu"][:nm] - ref["mu"][:nm])) / ref["mu"][0]
                s_prev = spectrum(prev["mu"][:nm], a, OMEGA)
                d_s = np.linalg.norm(s_prev - s_ref) / np.linalg.norm(s_ref)
                lines.append(
                    f"Reference: {reference} chi = {ref['chi']}, seed 0. Against chi ="
                    f" {prev['chi']}: max moment difference {d_mu:.1e} (of mu_0),"
                    f" S(q, w) difference {d_s:.1e} (relative L2 over w in [0, 4])."
                )
                lines.append("")
            if s_ref is not None:
                lo, hi = _edges(q)
                sigma = np.pi * a / nm
                w = np.linspace(0, 2 * a * W_PRIME, 20001)
                sw = spectrum(ref["mu"][:nm], a, w)
                inside = (w >= max(lo - sigma, 0)) & (w <= hi + sigma)
                frac = np.trapezoid(sw[inside], w[inside]) / np.trapezoid(sw, w)
                lines.append(
                    f"Physics check: edges {lo:.3f} and {hi:.3f}; with the kernel width"
                    f" pi a / N = {sigma:.3f}, {100 * frac:.1f}% of the reference weight"
                    f" lies in [{max(lo - sigma, 0):.3f}, {hi + sigma:.3f}]."
                )
                lines.append("")
            lines += [
                "Runs shorter than the reference are timing-only (errors `-`).",
                "",
                "| method | chi | seeds | steps | apply s | compress s | s / moment"
                f" | time for {nm} moments s | pool GB | max moment error | S error |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            points = []
            for method in METHODS:
                for chi in sorted({r["chi"] for r in tg if r["method"] == method}):
                    sub = [r for r in tg if r["method"] == method and r["chi"] == chi]
                    tim = [_timing(r) for r in sub]
                    per = median(t["per_moment_s"] for t in tim)
                    errs_mu, errs_s = [], []
                    if s_ref is not None:
                        for r in sub:
                            if len(r["mu"]) >= nm and r is not ref:
                                errs_mu.append(
                                    np.max(abs(r["mu"][:nm] - ref["mu"][:nm]))
                                    / ref["mu"][0]
                                )
                                errs_s.append(
                                    np.linalg.norm(
                                        spectrum(r["mu"][:nm], a, OMEGA) - s_ref
                                    )
                                    / np.linalg.norm(s_ref)
                                )
                    is_ref = any(r is ref for r in sub)
                    mu_cell = (
                        f"{max(errs_mu):.1e}" if errs_mu else ("ref" if is_ref else "-")
                    )
                    s_cell = (
                        f"{max(errs_s):.1e}" if errs_s else ("ref" if is_ref else "-")
                    )
                    oom_cell = " (OOM)" if any(r["oom"] for r in sub) else ""
                    lines.append(
                        f"| {method} | {chi} | {len(sub)} | {min(t['steps'] for t in tim)}"
                        f"{oom_cell} | {median(t['apply_s'] for t in tim):.3f}"
                        f" | {median(t['compress_s'] for t in tim):.3f} | {per:.3f}"
                        f" | {per * nm:.0f} | {max(t['pool_gb'] for t in tim):.1f}"
                        f" | {mu_cell} | {s_cell} |"
                    )
                    points.append(
                        (method, chi, per, per * nm, max(errs_s) if errs_s else None)
                    )
            lines.append("")
            tag = f"{model}_L{n}_{target}"
            fig, axes = plt.subplots(
                1,
                2 if s_ref is not None else 1,
                figsize=(11 if s_ref is not None else 6, 4),
                layout="constrained",
                squeeze=False,
            )
            ax = axes[0][0]
            for method in METHODS:
                pts = sorted((c, p) for m_, c, p, _, _ in points if m_ == method)
                if pts:
                    ax.loglog(
                        [c for c, _ in pts],
                        [p for _, p in pts],
                        color=COLORS[method],
                        marker=MARKERS[method],
                        linewidth=2,
                        markersize=7,
                        markeredgecolor="white",
                        label=method,
                    )
            ax.set_xlabel("chi")
            ax.set_ylabel("wall time per moment (s)")
            ax.set_title(f"{model} L = {n}, D = {D}", fontsize=10)
            _style(ax)
            ax.legend(frameon=False)
            if s_ref is not None:
                ax = axes[0][1]
                for method in METHODS:
                    pts = sorted(
                        (t, e, c)
                        for m_, c, _, t, e in points
                        if m_ == method and e is not None
                    )
                    if pts:
                        ax.loglog(
                            [t for t, _, _ in pts],
                            [e for _, e, _ in pts],
                            color=COLORS[method],
                            marker=MARKERS[method],
                            linewidth=2,
                            markersize=7,
                            markeredgecolor="white",
                            label=method,
                        )
                        for t, e, c in pts:
                            ax.annotate(
                                str(c),
                                (t, e),
                                textcoords="offset points",
                                xytext=(4, 4),
                                fontsize=7,
                                color="#52514e",
                            )
                ax.set_xlabel(f"time to {nm} moments (s)")
                ax.set_ylabel("S(q, w) relative L2 error (worst seed)")
                ax.set_title("time to spectrum at matched accuracy", fontsize=10)
                _style(ax)
                ax.legend(frameon=False)
            name = f"{tag}_time.png"
            fig.savefig(folder / name, dpi=150)
            plt.close(fig)
            lines += [f"![{name}]({name})", ""]
            if s_ref is not None:
                fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
                ax.plot(
                    OMEGA,
                    s_ref,
                    color="#2a78d6",
                    linewidth=2,
                    label=f"{reference} chi={ref['chi']}",
                )
                lo, hi = _edges(q)
                for x_, lab in ((lo, "lower edge"), (hi, "upper edge")):
                    ax.axvline(x_, color="#8a8984", linestyle=":", linewidth=1)
                    ax.text(
                        x_,
                        ax.get_ylim()[1] * 0.95,
                        f" {lab}",
                        fontsize=8,
                        color="#52514e",
                        va="top",
                    )
                ax.set_xlabel("w")
                ax.set_ylabel("S(q, w)")
                ax.set_title(
                    f"{model} L = {n}, q = {q / np.pi:.3f} pi, {nm} moments",
                    fontsize=10,
                )
                _style(ax)
                ax.legend(frameon=False)
                name = f"{tag}_spectrum.png"
                fig.savefig(folder / name, dpi=150)
                plt.close(fig)
                lines += [f"![{name}]({name})", ""]
    conclusion = folder / "conclusion.md"
    if conclusion.exists():
        lines += [conclusion.read_text().rstrip(), ""]
    (folder / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    app()
