"""Heisenberg operator evolution: one SRC sweep over ``U^dag O U`` against pairs.

Evolves ``O = Z`` on the central site of an open spin-1/2 Heisenberg chain,
``H = sum S_i . S_{i+1}``, by ``O <- U^dag O U`` with ``U`` an exact MPO of ``m``
first-order Trotter steps of ``dt``. Each step is compressed to ``chi`` by:

- ``stack``: ``src(Udag, O, U)``, one sweep (``stack-os``: oversampled 1.5x);
- ``pair-trunc``: ``src(src(Udag, O), U)``, both truncated to ``chi``;
- ``pair-wide``: the same with the intermediate at ``D * chi``, effectively exact.

Commands:

- ``check``: the ``U^dag U = 1`` and sandwich sanity checks, and the gate
  reference against the dense evolution;
- ``reference``: the converged reference for chains too long to be dense, saved
  to disk for ``run``;
- ``run``: every method and ``chi`` for one ``(n, D)``, one JSON line per step;
- ``report``: merge the runs into ``results.jsonl``, a Markdown summary and plots.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
from contextlib import contextmanager
from functools import cache
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cyclopts
import numpy as np
from scipy.linalg import expm

import src_method
from src_method import src

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Stack against pairwise SRC for Heisenberg operator evolution.")

HERE = Path(__file__).parent
OUT = HERE / "leonardo"
DT = 0.1
FLOOR = 1e-7  # Inner-product distances bottom out near sqrt(eps) ~ 1e-8.
U_TOL = 1e-12
METHODS = ("stack", "stack-os", "pair-trunc", "pair-wide")
CHIS = (16, 32, 64, 128, 256)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

Train = list[np.ndarray]


# ---------------------------------------------------------------------------
# Trains: "full" form pads the boundary sites to (l, r, u, d) with unit bonds
# ---------------------------------------------------------------------------


def to_full(train: Train) -> Train:
    """Pad the boundary sites of a quimb-layout MPO to ``(l, r, u, d)``."""
    return [train[0][None], *train[1:-1], train[-1][:, None]]


def to_quimb(full: Train) -> Train:
    """Drop the outer bonds of a full-form MPO."""
    return [full[0][0], *full[1:-1], full[-1][:, 0]]


def dagger(train: Train) -> Train:
    """Site-wise adjoint: conjugate and swap ``u`` with ``d``, the last two axes."""
    return [t.conj().swapaxes(-1, -2) for t in train]


def bond(train: Train) -> int:
    """Largest bond dimension of a quimb-layout MPO."""
    return max(t.shape[0] for t in train[1:])


def mpo_mul(a: Train, b: Train) -> Train:
    """Exact product ``a . b`` of full-form MPOs: ``a``'s ``d`` meets ``b``'s ``u``."""
    out = []
    for x, y in zip(a, b, strict=True):
        t = np.einsum("abux,cdxv->acbduv", x, y)
        s = t.shape
        out.append(t.reshape(s[0] * s[1], s[2] * s[3], s[4], s[5]))
    return out


def svd_compress(full: Train, tol: float, max_bond: int | None = None) -> Train:
    """Canonical SVD truncation, dropping ``s < tol * s_max`` at every bond."""
    sites = list(full)
    for j in range(len(sites) - 1):
        l, r, u, d = sites[j].shape
        Q, R = np.linalg.qr(sites[j].transpose(0, 2, 3, 1).reshape(-1, r))
        sites[j] = Q.reshape(l, u, d, -1).transpose(0, 3, 1, 2)
        sites[j + 1] = np.einsum("kr,rsud->ksud", R, sites[j + 1])
    for j in range(len(sites) - 1, 0, -1):
        l, r, u, d = sites[j].shape
        U, S, Vh = np.linalg.svd(sites[j].reshape(l, -1), full_matrices=False)
        k = int((tol * S[0] < S).sum())
        if max_bond is not None:
            k = min(k, max_bond)
        sites[j] = Vh[:k].reshape(k, r, u, d)
        sites[j - 1] = np.einsum("alud,lk->akud", sites[j - 1], U[:, :k] * S[:k])
    return sites


def heisenberg_gate(dt: float) -> np.ndarray:
    """``exp(-i dt S.S)`` as a 4x4 matrix, rows ``(u1 u2)``, columns ``(d1 d2)``."""
    h = (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) / 4
    return expm(-1j * dt * h)


def gate_layer(n: int, parity: int, gate: np.ndarray) -> Train:
    """Full-form MPO of ``gate`` on the bonds ``(i, i + 1)`` with ``i % 2 == parity``."""
    sites = [I2[None, None] for _ in range(n)]
    split = gate.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    U, S, Vh = np.linalg.svd(split)
    k = int((S > 1e-14 * S[0]).sum())
    left = (U[:, :k] * np.sqrt(S[:k])).reshape(2, 2, k).transpose(2, 0, 1)
    right = (np.sqrt(S[:k])[:, None] * Vh[:k]).reshape(k, 2, 2)
    for i in range(parity, n - 1, 2):
        sites[i], sites[i + 1] = left[None], right[:, None]
    return sites


def layer_parities(m: int) -> list[int]:
    """Parities of the gate layers of ``U_m``, from the one adjacent to ``O``.

    ``U_m = (U_odd U_even)^m``, so in ``U_m^dag O U_m`` the odd layer acts first.
    """
    return [1, 0] * m


@cache
def trotter(n: int, m: int) -> tuple[Train, Train]:
    """``U_m = (U_odd U_even)^m`` and its adjoint, quimb layout.

    ``U_1`` is the exact product of the two layers (bond 4); higher powers are
    contracted exactly and recompressed at a relative tolerance of 1e-12.
    """
    gate = heisenberg_gate(DT)
    step = mpo_mul(gate_layer(n, 1, gate), gate_layer(n, 0, gate))
    U = svd_compress(step, U_TOL)
    for _ in range(m - 1):
        U = svd_compress(mpo_mul(U, step), U_TOL)
    U = to_quimb(U)
    return U, dagger(U)


def z_center(n: int) -> Train:
    """``Z`` on site ``n // 2`` as a bond-1 MPO."""
    full = [I2[None, None] for _ in range(n)]
    full[n // 2] = Z[None, None]
    return to_quimb(full)


# ---------------------------------------------------------------------------
# Backends, dense algebra and inner products
# ---------------------------------------------------------------------------


@cache
def xp_of(device: str) -> ModuleType:
    """The array module of the analysis: cupy on the GPU, numpy otherwise."""
    if device == "gpu":
        import cupy  # noqa: PLC0415  (only the GPU runs need it)

        return cupy
    return np


@contextmanager
def analysis_pool(device: str) -> Iterator[None]:
    """Run analysis on a separate CuPy pool, so the default one only sees `src`."""
    if device != "gpu":
        yield
        return
    cp = xp_of(device)
    pool = _analysis_pool()
    with cp.cuda.using_allocator(pool.malloc):
        yield


@cache
def _analysis_pool() -> Any:  # noqa: ANN401  (a cupy MemoryPool)
    return xp_of("gpu").cuda.MemoryPool()


def src_pool_reset(device: str) -> None:
    """Free the blocks cached by the default CuPy pool before a run."""
    if device == "gpu":
        xp_of(device).get_default_memory_pool().free_all_blocks()


def src_pool_bytes(device: str) -> int:
    """Bytes held by the default pool: its high-water mark since the last reset."""
    if device != "gpu":
        return 0
    return int(xp_of(device).get_default_memory_pool().total_bytes())


def to_dense(train: Train, xp: ModuleType) -> Any:  # noqa: ANN401
    """The ``(2^n, 2^n)`` matrix of a quimb-layout MPO, rows on ``u``."""
    full = [xp.asarray(t) for t in to_full(train)]
    M = full[0][0]  # (r, u, d)
    M = xp.moveaxis(M, 0, -1)  # (rows, cols, r)
    for W in full[1:]:
        t = xp.einsum("abr,rsud->aubds", M, W)
        s = t.shape
        M = t.reshape(s[0] * s[1], s[2] * s[3], s[4])
    return M[..., 0]


def inner(a: Train, b: Train, xp: ModuleType) -> complex:
    """``Tr(a^dag b) / 2^n`` of two quimb-layout MPOs."""
    env = xp.ones((1, 1), dtype=complex)
    for x, y in zip(to_full(a), to_full(b), strict=True):
        tmp = xp.tensordot(env, xp.conj(xp.asarray(x)), axes=(0, 0))  # (b, c, u, d)
        env = xp.tensordot(tmp, xp.asarray(y), axes=([0, 2, 3], [0, 2, 3])) / 2
    return complex(env[0, 0])


def distance(ref: Train, got: Train, xp: ModuleType) -> float:
    """``||got - ref|| / ||ref||`` from inner products (floor ~ sqrt(eps))."""
    rr, gg, rg = inner(ref, ref, xp), inner(got, got, xp), inner(ref, got, xp)
    return float(np.sqrt(max(rr.real + gg.real - 2 * rg.real, 0.0) / rr.real))


def correlator(train: Train, xp: ModuleType) -> float:
    """``C = Tr(O Z_c) / 2^n``."""
    return inner(z_center(len(train)), train, xp).real


# ---------------------------------------------------------------------------
# Gate-by-gate reference
# ---------------------------------------------------------------------------


class GateEvolution:
    """``O <- U_m^dag O U_m`` gate by gate, with canonical SVD truncation.

    Every two-site gate conjugates the operator, ``O <- g^dag O g``, a unitary on
    the vectorized operator, so keeping an orthogonality centre on the gated pair
    makes each truncation the optimal one at that bond. The centre sweeps right
    through one layer and left through the next. This is the exact contraction of
    ``U_m^dag O U_m`` with SVD compression, in pieces small enough for bond 1024:
    contracting the whole sandwich first would need bond ``D^2 chi``.

    Args:
        n: Number of sites.
        m: Trotter steps per ``U_m``.
        max_bond: Bond cap of the truncation.
        cutoff: Relative singular-value cutoff, ``s < cutoff * s_max`` dropped.
        device: ``"gpu"`` or ``"cpu"``.
        start: The operator to start from (quimb layout), ``Z_c`` by default.
        discarded: Discarded weight carried over from earlier steps.
    """

    def __init__(
        self,
        n: int,
        m: int,
        max_bond: int,
        cutoff: float,
        device: str,
        start: Train | None = None,
        discarded: float = 0.0,
    ) -> None:
        self.xp = xp = xp_of(device)
        self.max_bond, self.cutoff, self.m = max_bond, cutoff, m
        self.gate = xp.asarray(heisenberg_gate(DT))
        start = z_center(n) if start is None else start
        self.sites = [xp.asarray(t) for t in to_full(start)]
        self.centre = 0
        self.discarded = discarded
        self._move(len(self.sites) - 1)
        self._move(0)

    def _move(self, to: int) -> None:
        xp, s = self.xp, self.sites
        while self.centre < to:
            j = self.centre
            l, r, u, d = s[j].shape
            Q, R = xp.linalg.qr(s[j].transpose(0, 2, 3, 1).reshape(-1, r))
            s[j] = Q.reshape(l, u, d, -1).transpose(0, 3, 1, 2)
            s[j + 1] = xp.tensordot(R, s[j + 1], axes=(1, 0))
            self.centre += 1
        while self.centre > to:
            j = self.centre
            l, r, u, d = s[j].shape
            Q, R = xp.linalg.qr(s[j].reshape(l, -1).T)
            s[j] = Q.T.reshape(-1, r, u, d)
            s[j - 1] = xp.tensordot(s[j - 1], R.T, axes=(1, 0)).transpose(0, 3, 1, 2)
            self.centre -= 1

    def _gate(self, i: int, rightwards: bool) -> None:
        xp, s = self.xp, self.sites
        a, b = s[i], s[i + 1]
        l, r = a.shape[0], b.shape[1]
        theta = xp.tensordot(a, b, axes=(1, 0))  # (l, u1, d1, r, u2, d2)
        theta = theta.transpose(0, 3, 1, 4, 2, 5).reshape(l, r, 4, 4)
        theta = self.gate.conj().T @ theta @ self.gate
        theta = theta.reshape(l, r, 2, 2, 2, 2).transpose(0, 2, 4, 3, 5, 1)
        U, S, Vh = xp.linalg.svd(theta.reshape(l * 4, 4 * r), full_matrices=False)
        S_host = xp.asnumpy(S) if hasattr(xp, "asnumpy") else S
        k = min(self.max_bond, int((self.cutoff * S_host[0] < S_host).sum()))
        self.discarded += float((S_host[k:] ** 2).sum() / (S_host**2).sum())
        U, S, Vh = U[:, :k], S[:k], Vh[:k]
        if rightwards:
            Vh = S[:, None] * Vh
        else:
            U = U * S
        s[i] = U.reshape(l, 2, 2, k).transpose(0, 3, 1, 2)
        s[i + 1] = Vh.reshape(k, 2, 2, r).transpose(0, 3, 1, 2)
        self.centre = i + 1 if rightwards else i

    def step(self) -> None:
        """Apply one ``U_m^dag . U_m``."""
        n = len(self.sites)
        for parity in layer_parities(self.m):
            bonds = list(range(parity, n - 1, 2))
            rightwards = self.centre <= n // 2
            for i in bonds if rightwards else reversed(bonds):
                self._move(i if rightwards else i + 1)
                self._gate(i, rightwards)

    def train(self) -> Train:
        """The current operator, host-side and in quimb layout."""
        xp = self.xp
        host = [xp.asnumpy(t) if hasattr(xp, "asnumpy") else t for t in self.sites]
        return to_quimb(host)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def make_step(
    method: str, chi: int, U: Train, Udag: Train, device: str
) -> Callable[[Train, int], Train]:
    """One ``O <- U^dag O U`` by ``method``, seeded per step."""
    D = bond(U)
    opts = {"cutoff": 0.0, "dtype": np.complex128, "device": device}

    def stack(O: Train, seed: int) -> Train:
        return src(Udag, O, U, chi_out=chi, oversample=1.0, seed=seed, **opts)

    def stack_os(O: Train, seed: int) -> Train:
        return src(Udag, O, U, chi_out=chi, oversample=None, seed=seed, **opts)

    def pair(mid: int) -> Callable[[Train, int], Train]:
        def run(O: Train, seed: int) -> Train:
            half = src(Udag, O, chi_out=mid, oversample=1.0, seed=seed, **opts)
            return src(half, U, chi_out=chi, oversample=1.0, seed=seed + 1, **opts)

        return run

    return {
        "stack": stack,
        "stack-os": stack_os,
        "pair-trunc": pair(chi),
        "pair-wide": pair(D * chi),
    }[method]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def provenance(device: str) -> dict[str, Any]:
    """Commit, contraction-path setting, versions and hardware of this run."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    kernels = Path(src_method.__file__).with_name("_kernels.py").read_text()
    meta: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "kernels_optimize": "dp"
        if re.search(r"""optimize\s*=\s*["']dp["']""", kernels)
        else "default (opt_einsum 'auto')",
        "src_method": src_method.__version__,
        "numpy": np.__version__,
        "opt_einsum": version("opt_einsum"),
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "device": device,
    }
    if device == "gpu":
        cp = xp_of(device)
        meta["cupy"] = cp.__version__
        meta["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return meta


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command
def check(device: str = "gpu", output: Path = OUT / "checks.json") -> None:
    """Sanity checks; fail loudly if any is off.

    - ``U^dag U`` is the identity on 4 sites (m = 1, 2);
    - ``U_m`` matches the dense gate product on 6 sites;
    - every method at ``chi = 128`` (above the exact bond 64) matches the dense
      sandwich on 6 sites, 5 steps;
    - the gate reference matches the dense evolution on 8 sites, 20 steps.

    Args:
        device: ``"gpu"`` or ``"cpu"``.
        output: JSON file for the measured deviations.
    """
    xp = xp_of(device)
    out: dict[str, Any] = {"meta": provenance(device)}
    gate = heisenberg_gate(DT)
    for m in (1, 2):
        U, Udag = trotter(4, m)
        eye = to_dense(to_quimb(mpo_mul(to_full(Udag), to_full(U))), np)
        out[f"udag_u_n4_m{m}"] = float(np.abs(eye - np.eye(16)).max())

        n = 6
        U, Udag = trotter(n, m)

        def kron_layer(parity: int, n: int = n) -> np.ndarray:
            ops, i = [], 0
            while i < n:
                pair = i % 2 == parity and i + 1 < n
                ops.append(gate if pair else I2)
                i += 2 if pair else 1
            out = np.eye(1)
            for o in ops:
                out = np.kron(out, o)
            return out

        exact = np.linalg.matrix_power(kron_layer(1) @ kron_layer(0), m)
        Ud = to_dense(U, np)
        out[f"u_dense_n6_m{m}"] = float(np.abs(Ud - exact).max())
        ref = to_dense(z_center(n), np)
        Os = {meth: z_center(n) for meth in METHODS}
        for t in range(5):
            ref = Ud.conj().T @ ref @ Ud
            for meth in METHODS:
                Os[meth] = make_step(meth, 128, U, Udag, device)(Os[meth], t)
        for meth in METHODS:
            got = to_dense(Os[meth], np)
            out[f"sandwich_n6_m{m}_{meth}"] = float(
                np.linalg.norm(got - ref) / np.linalg.norm(ref)
            )
        # The inner-product distance and C(t) against their dense values, on a
        # truncated pair far from the floor.
        a = make_step("stack", 4, U, Udag, device)(Os["stack"], 99)
        b = Os["pair-wide"]
        da, db = to_dense(a, np), to_dense(b, np)
        dense_dist = np.linalg.norm(da - db) / np.linalg.norm(db)
        out[f"inner_vs_dense_n6_m{m}"] = abs(distance(b, a, xp) - dense_dist)
        dense_c = np.vdot(to_dense(z_center(n), np), db).real / 2**n
        out[f"correlator_vs_dense_n6_m{m}"] = abs(correlator(b, xp) - dense_c)

        n = 8
        U, _ = trotter(n, m)
        Ud = to_dense(U, np)
        ref = to_dense(z_center(n), np)
        evo = GateEvolution(n, m, 1024, 1e-12, device)
        for _ in range(20):
            ref = Ud.conj().T @ ref @ Ud
            evo.step()
        got = to_dense(evo.train(), np)
        out[f"gate_ref_n8_m{m}"] = float(
            np.linalg.norm(got - ref) / np.linalg.norm(ref)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    # U and its adjoint are exact to rounding; everything SRC touches, to the floor.
    tight = ("udag_u", "u_dense", "correlator")
    bad = {
        k: v
        for k, v in out.items()
        if k != "meta" and v > (1e-12 if k.startswith(tight) else FLOOR)
    }
    for k, v in out.items():
        if k != "meta":
            logger.info("check %s: %.3e", k, v)
    if bad:
        msg = f"checks failed: {bad}"
        raise SystemExit(msg)


def ref_dir(n: int, m: int) -> Path:
    """Scratch folder of the saved reference trajectory."""
    root = Path(os.environ.get("CINECA_SCRATCH", HERE / "leonardo"))
    return root / "heisenberg-ref" / f"n{n}_m{m}"


@app.command
def reference(
    n: int = 24,
    m: int = 2,
    steps: int = 20,
    max_bond: int = 4 * max(CHIS),
    cutoff: float = 1e-12,
    device: str = "gpu",
    budget_minutes: float = 25.0,
) -> None:
    """Gate-by-gate reference at ``max_bond``, checked against ``max_bond / 2``.

    Saves both operators after every step to `ref_dir` and writes one JSON line
    per step with the convergence distance, ``C(t)`` and the bonds. Resumes from
    the last saved step and stops before a step could overrun ``budget_minutes``,
    so a long reference spans several short jobs.

    Args:
        n: Number of sites.
        m: Trotter steps per ``U_m``.
        steps: Number of applications of ``U_m``.
        max_bond: Bond cap of the reference.
        cutoff: Relative singular-value cutoff.
        device: ``"gpu"`` or ``"cpu"``.
        budget_minutes: Wall-time budget of this invocation.
    """
    job_start = perf_counter()
    xp = xp_of(device)
    folder = ref_dir(n, m)
    (folder / "half").mkdir(parents=True, exist_ok=True)
    path = OUT / f"reference_n{n}_m{m}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    )
    done = [
        r
        for r in rows
        if "meta" not in r and (folder / "half" / f"step{r['step']:02d}.npz").exists()
    ]
    if done:
        last = done[-1]
        t0 = last["step"]
        fine = GateEvolution(
            n, m, max_bond, cutoff, device, load_train(folder, t0, n), last["discarded"]
        )
        coarse = GateEvolution(
            n,
            m,
            max_bond // 2,
            cutoff,
            device,
            load_train(folder / "half", t0, n),
            last["discarded_half"],
        )
        logger.info("resuming the reference after step %d", t0)
    else:
        t0 = 0
        fine = GateEvolution(n, m, max_bond, cutoff, device)
        coarse = GateEvolution(n, m, max_bond // 2, cutoff, device)
    slowest = 0.0
    with path.open("w") as f:
        f.write(json.dumps({"meta": provenance(device)}) + "\n")
        for r in done:
            f.write(json.dumps(r) + "\n")
        for t in range(t0 + 1, steps + 1):
            elapsed = perf_counter() - job_start
            if elapsed + 1.5 * slowest > 60 * budget_minutes:
                logger.info("budget reached before step %d; resubmit to resume", t)
                return
            start = perf_counter()
            fine.step()
            coarse.step()
            seconds = perf_counter() - start
            R, R2 = fine.train(), coarse.train()
            np.savez(folder / f"step{t:02d}.npz", *R)
            np.savez(folder / "half" / f"step{t:02d}.npz", *R2)
            row = {
                "n": n,
                "m": m,
                "step": t,
                "time": round(t * m * DT, 10),
                "bond": bond(R),
                "bond_half": bond(R2),
                "max_bond": max_bond,
                "discarded": fine.discarded,
                "discarded_half": coarse.discarded,
                "converged_distance": distance(R, R2, xp),
                "norm": inner(R, R, xp).real,
                "C": correlator(R, xp),
                "C_half": correlator(R2, xp),
                "seconds": seconds,
            }
            slowest = max(slowest, perf_counter() - start)
            f.write(json.dumps(row) + "\n")
            f.flush()
            logger.info("reference: %s", row)
    logger.info("reference complete")


def load_train(folder: Path, t: int, n: int) -> Train:
    """One saved step of a reference trajectory."""
    with np.load(folder / f"step{t:02d}.npz") as z:
        return [z[f"arr_{i}"] for i in range(n)]


def load_reference(n: int, m: int, steps: int) -> list[Train]:
    """The saved reference trajectory of `reference`."""
    return [load_train(ref_dir(n, m), t, n) for t in range(1, steps + 1)]


@app.command
def run(
    n: int = 12,
    m: int = 2,
    steps: int = 20,
    chis: tuple[int, ...] = CHIS,
    methods: tuple[str, ...] = METHODS,
    device: str = "gpu",
    seed: int = 0,
) -> None:
    """Every method and ``chi`` for one ``(n, m)``; one JSON line per step.

    ``n <= 14`` uses the dense reference ``U^t^dag Z U^t``; longer chains load
    the trajectory saved by `reference`.

    Args:
        n: Number of sites.
        m: Trotter steps per ``U_m`` (``D = 4`` for 1, up to 16 for 2).
        steps: Number of applications of ``U_m``.
        chis: Output bond dimensions; those at or above ``4^(n // 2)`` are skipped.
        methods: Subset of the methods.
        device: ``"gpu"`` or ``"cpu"``.
        seed: Base seed; step ``t`` uses ``seed + 2 t`` (and ``+ 1`` for the
            second half of a pair).
    """
    xp = xp_of(device)
    U, Udag = trotter(n, m)
    D = bond(U)
    dense = n <= 14
    exact_cap = 4 ** (n // 2)
    chis = tuple(c for c in chis if c < exact_cap)
    logger.info("n=%d m=%d D=%d chis=%s dense=%s", n, m, D, chis, dense)

    with analysis_pool(device):
        if dense:
            Ud = xp.asarray(to_dense(U, np))
            Zc = to_dense(z_center(n), xp)
            R = Zc
            refs, c_refs = [], []
            for _ in range(steps):
                R = Ud.conj().T @ R @ Ud
                refs.append(xp.asnumpy(R) if hasattr(xp, "asnumpy") else R.copy())
                c_refs.append(float(xp.real(xp.vdot(Zc, R))) / 2**n)
            del Ud, Zc, R
        else:
            refs = load_reference(n, m, steps)
            c_refs = [correlator(R, xp) for R in refs]

    # The first SRC calls of a job compile CuPy kernels: warm up every method.
    for method in methods:
        make_step(method, min(chis), U, Udag, device)(z_center(n), 0)

    meta = provenance(device) | {"D": D, "chis": chis, "methods": methods}
    path = OUT / f"run_n{n}_m{m}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for chi in chis:
            for method in methods:
                step = make_step(method, chi, U, Udag, device)
                O = z_center(n)
                src_pool_reset(device)
                peak = 0
                for t in range(1, steps + 1):
                    start = perf_counter()
                    O = step(O, seed + 2 * t)
                    seconds = perf_counter() - start
                    peak = max(peak, src_pool_bytes(device))
                    with analysis_pool(device):
                        if dense:
                            ref = xp.asarray(refs[t - 1])
                            diff = to_dense(O, xp) - ref
                            err = float(xp.linalg.norm(diff) / xp.linalg.norm(ref))
                            del ref, diff
                        else:
                            err = distance(refs[t - 1], O, xp)
                        c = correlator(O, xp)
                    row = {
                        "n": n,
                        "m": m,
                        "D": D,
                        "chi": chi,
                        "method": method,
                        "step": t,
                        "time": round(t * m * DT, 10),
                        "error": err,
                        "error_kind": "dense" if dense else "inner-product",
                        "at_floor": (not dense) and err < FLOOR,
                        "C": c,
                        "C_ref": c_refs[t - 1],
                        "C_error": abs(c - c_refs[t - 1]),
                        "bond": bond(O),
                        "seconds": seconds,
                        "peak_pool_bytes": peak,
                    }
                    f.write(json.dumps(row) + "\n")
                f.flush()
                logger.info(
                    "chi=%d %s: err=%.3e C_err=%.3e median %.3f s, pool %.2f GB",
                    chi,
                    method,
                    err,
                    row["C_error"],
                    median(r for r in _times(path, chi, method)),
                    peak / 1e9,
                )


def _times(path: Path, chi: int, method: str) -> list[float]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [
        r["seconds"]
        for r in rows
        if "meta" not in r and r["chi"] == chi and r["method"] == method
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Categorical slots 1-4 of the reference palette, fixed per method.
COLORS = {
    "stack": "#2a78d6",
    "stack-os": "#eb6834",
    "pair-trunc": "#1baf7a",
    "pair-wide": "#eda100",
}
MARKERS = {"stack": "o", "stack-os": "s", "pair-trunc": "^", "pair-wide": "D"}
# pair-wide is drawn first and larger: stack often lies right on top of it.
SIZES = {"stack": 6, "stack-os": 6, "pair-trunc": 7, "pair-wide": 10}


def _read_runs(folder: Path) -> tuple[list[dict], dict[str, dict]]:
    rows, metas = [], {}
    for path in sorted(folder.glob("run_n*_m*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if "meta" in rec:
                metas[path.stem] = rec["meta"]
            else:
                rows.append(rec)
    return rows, metas


def _plot(rows: list[dict], n: int, m: int, D: int, folder: Path) -> list[str]:
    import matplotlib as mpl  # noqa: PLC0415  (only the report needs it)

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    final = [
        r
        for r in rows
        if r["n"] == n and r["m"] == m and r["step"] == max(x["step"] for x in rows)
    ]
    names = []
    for kind, xlabel in (("chi", "chi"), ("time", "median time per step (s)")):
        fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
        for method in reversed(METHODS):
            pts = sorted(
                (r for r in final if r["method"] == method), key=lambda r: r["chi"]
            )
            if not pts:
                continue
            xs = (
                [r["chi"] for r in pts]
                if kind == "chi"
                else [_median_time(rows, n, m, r["chi"], method) for r in pts]
            )
            ax.plot(
                xs,
                [r["error"] for r in pts],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=SIZES[method],
                markeredgecolor="white",
                markeredgewidth=1.2,
                linewidth=2,
                label=method,
            )
        handles, labels = ax.get_legend_handles_labels()
        if n > 14 and min(r["error"] for r in final) < 10 * FLOOR:
            ax.axhline(FLOOR, color="#8a8984", linewidth=1, linestyle=":")
            ax.text(
                ax.get_xlim()[0],
                FLOOR,
                " inner-product floor",
                va="bottom",
                fontsize=8,
                color="#52514e",
            )
        ax.set_xscale("log", base=2 if kind == "chi" else 10)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("relative Frobenius error at final step")
        ax.set_title(
            f"n={n}, D={D} (m={m}), T={max(r['step'] for r in rows)} steps", fontsize=10
        )
        ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles[::-1], labels[::-1], frameon=False)
        name = f"n{n}_D{D}_error_vs_{kind}.png"
        fig.savefig(folder / name, dpi=150)
        plt.close(fig)
        names.append(name)

    chi = 64
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    for method in reversed(METHODS):
        pts = sorted(
            (
                r
                for r in rows
                if r["n"] == n
                and r["m"] == m
                and r["chi"] == chi
                and r["method"] == method
            ),
            key=lambda r: r["step"],
        )
        ax.plot(
            [r["time"] for r in pts],
            [max(r["error"], 1e-9) for r in pts],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=SIZES[method] - 2,
            markeredgecolor="white",
            markeredgewidth=1,
            linewidth=2,
            label=method,
        )
    handles, labels = ax.get_legend_handles_labels()
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title(f"n={n}, D={D} (m={m}), chi={chi}", fontsize=10)
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles[::-1], labels[::-1], frameon=False)
    name = f"n{n}_D{D}_error_vs_t_chi{chi}.png"
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)
    names.append(name)
    return names


def _median_time(rows: list[dict], n: int, m: int, chi: int, method: str) -> float:
    return median(
        r["seconds"]
        for r in rows
        if r["n"] == n and r["m"] == m and r["chi"] == chi and r["method"] == method
    )


@app.command
def report(folder: Path = OUT) -> None:
    """Merge the runs into ``results.jsonl`` and write ``summary.md`` with plots.

    A hand-written ``conclusion.md`` in ``folder``, if present, is appended.

    Args:
        folder: Folder with the ``run_*.jsonl`` and ``reference_*.jsonl`` files.
    """
    rows, metas = _read_runs(folder)
    with (folder / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lines = ["# Heisenberg operator evolution: stack against pairwise SRC", ""]
    lines += [
        "## Provenance",
        "",
        "| run | "
        + " | ".join(("commit", "kernels optimize", "cupy", "numpy", "gpu", "job"))
        + " |",
        "|---" * 7 + "|",
    ]
    for name, meta in metas.items():
        lines.append(
            f"| {name} | {meta['commit'][:10]}{' (dirty src)' if meta['src_dirty'] else ''}"
            f" | {meta['kernels_optimize']} | {meta.get('cupy', '-')} | {meta['numpy']}"
            f" | {meta.get('gpu', '-')} | {meta['slurm_job']} |"
        )
    lines.append("")

    refs = sorted(folder.glob("reference_n*_m*.jsonl"))
    if refs:
        lines += [
            "## Reference convergence (n = 24)",
            "",
            "`distance`: between the references at bond 4 and 2 chi_max, from inner"
            " products (0 means below the ~1e-8 floor). `bound`: a rigorous bound on"
            " the truncation error of each, `sqrt(N_gates * sum of discarded"
            " weights)`, since every gate is unitary on the vectorized operator.",
            "",
            "| m | step | bond 4 chi_max | bond 2 chi_max | distance | bound 4 chi_max"
            " | bound 2 chi_max | C(t) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for path in refs:
            recs = [json.loads(x) for x in path.read_text().splitlines()]
            for r in recs:
                if "meta" in r or r["step"] % 5:
                    continue
                gates = r["step"] * r["m"] * (r["n"] - 1)
                lines.append(
                    f"| {r['m']} | {r['step']} | {r['bond']} | {r['bond_half']}"
                    f" | {r['converged_distance']:.1e}"
                    f" | {np.sqrt(gates * r['discarded']):.1e}"
                    f" | {np.sqrt(gates * r['discarded_half']):.1e} | {r['C']:.6f} |"
                )
        lines.append("")

    T = max(r["step"] for r in rows)
    for n, m in sorted({(r["n"], r["m"]) for r in rows}):
        D = next(r["D"] for r in rows if r["n"] == n and r["m"] == m)
        sub = [r for r in rows if r["n"] == n and r["m"] == m and r["step"] == T]
        lines += [
            f"## n = {n}, D = {D} (m = {m}), t = {T * m * DT:g}",
            "",
            "Error at the final step (`*`: at the inner-product floor, < 1e-7), "
            "median seconds per step, peak CuPy pool.",
            "",
            "| chi | method | error | C(T) error | bond | s/step | pool GB |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(sub, key=lambda r: (r["chi"], METHODS.index(r["method"]))):
            flag = "*" if r["at_floor"] else ""
            lines.append(
                f"| {r['chi']} | {r['method']} | {r['error']:.2e}{flag}"
                f" | {r['C_error']:.1e} | {r['bond']}"
                f" | {_median_time(rows, n, m, r['chi'], r['method']):.3f}"
                f" | {r['peak_pool_bytes'] / 1e9:.2f} |"
            )
        lines.append("")
        for name in _plot(rows, n, m, D, folder):
            lines += [f"![{name}]({name})", ""]

    conclusion = folder / "conclusion.md"
    if conclusion.exists():
        lines += [conclusion.read_text().rstrip(), ""]
    (folder / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    app()
