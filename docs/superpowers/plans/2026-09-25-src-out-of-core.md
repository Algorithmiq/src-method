# Out-of-core SRC Sweep on One GPU (Phase 1): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one `src` sweep run when the stack does not fit on the GPU, by streaming the input cores, batching every contraction to a memory budget and keeping the environments on the GPU, in host memory or on local disk.

**Architecture:** The sweep in `_sweep.py` becomes a driver over four parts: `_sites.py` reads the padded cores of one site at a time (with a background prefetch); `_kernels.py` holds the einsum equations and the four contractions restricted to a batch, plus a peak-memory estimate walked from the `opt_einsum` path; `_plan.py` turns `Resources` into byte budgets and plans, per site, the batch sizes and the tier of each environment; `_store.py` keeps the environments on the tier the plan chose, moving off-device batches through page-locked buffers with a writer and a reader thread. The mathematics and the random draws are unchanged.

**Tech Stack:** Python 3.11-3.14, NumPy (CuPy via `src_method.utils._backend`), `opt_einsum`, `structlog`, `pytest`, `cyclopts` (benchmark), `uv`, `ruff` through `prek`, GitButler (`but`) for commits.

**Spec:** `docs/superpowers/specs/2026-09-25-src-out-of-core-design.md` (read it first; this plan argues from it).

## Global Constraints

- Work on branch `feat/src-out-of-core`, stacked on `feat/src-stack`; the spec is committed there. The repository uses GitButler: commit with `but commit`, never `git add`/`git commit`.
- `from __future__ import annotations` at the top of every module under `src/`; type hints everywhere; Google-style docstrings without types on every public function, class and module.
- `ruff` (formatter at 88 columns, `E501` past 120) is the source of truth for style; run `uv run ruff format` rather than hand-formatting.
- Algorithms get the array module as `xp` and go through `src_method.utils._backend`; never import `cupy` in library code outside `_backend.py`. Random draws stay host-side via `default_rng(seed)`, one full `omega` per site, as today.
- Log with `structlog`, never `print`.
- `src`, `apply` and `compress` are pure: never mutate input arrays.
- `filterwarnings = ["error"]`: a stray warning (including `ResourceWarning` from threads or files) fails the suite.
- Commits: `<type>(<optional scope>): <gitmoji> <description>`, ending with the trailer `Assisted-by: Pi:claude-opus-5-5`, no `Co-authored-by`.
- Before pushing: `uv run prek run --all-files` and `uv run pytest` must pass.
- Reference problem (spec): 50 sites, physical legs of 4, `D_N = D_V = D_U = 4`, `D_M = 4000`, `chi_out = 2000`, complex128, one A100-40GB, at least 300 GB of host memory, node-local NVMe.
- Budgets: GPU default is free device memory plus CuPy pool free bytes minus `max(10%, 1 GiB)`; host default is `MemAvailable` minus 10%; disk is free space minus 5%; `"36GB"` is `36 * 10**9` bytes and `"36GiB"` is `36 * 2**30`.
- Out of scope (spec non-goals): distribution over several GPUs or nodes, any change to the mathematics, oversampling, writing the output to disk, caching small layers, recomputing environments.

## Refinements of the spec

The code below was prototyped and its tests run on the CPU backend (every task leaves the suite green). Five points refine the spec; the executor should not "fix" them back:

1. **Host-to-device copies run on the compute stream**, from page-locked buffers, not on the copy stream. CuPy's pool reuses a block freed by one stream for the next allocation on that stream, so a copy on another stream could overwrite memory that queued kernels still read. On the compute stream the copy is ordered with those kernels. Device-to-host copies stay on the copy stream (after an event on the compute stream), and the reading and staging, the slow part, still overlaps compute on the background threads.
2. **Batch sizes come from a binary search** over batch sizes whose peak is measured by walking the `opt_einsum` path, instead of the linear formula; no `memory_limit` is passed to `opt_einsum`, because a path that cannot meet the limit degrades into one large einsum.
3. **The working dtype** of environments and outputs is `np.result_type(dtype, *core dtypes)`, as the contractions already promote today.
4. **Lazy sites in bra stacks**: `transpose_mpo` wraps sites without `swapaxes` (zarr, HDF5) in `SwappedLegs`, so that the bra form works on them too.
5. **Integration tests compare the dense operators**, not the cores: batching changes the rounding, and with it the cores of an ill-conditioned sketch, not the operator.

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/src_method/utils/_backend.py` | modify | streams and events (null on the host), page-locked buffers, async copies, device and host memory queries, pool cap |
| `src/src_method/utils/__init__.py` | modify | export the new helpers |
| `src/src_method/_tensor_train.py` | modify | `padded_shape`, `pad_site`, `SwappedLegs`; `pad` and `transpose_mpo` built on them |
| `src/src_method/_kernels.py` | create | `equations`, `_Contractions`, `SiteKernels`, `peak_elements` |
| `src/src_method/_plan.py` | create | `parse_size`, `Resources`, `Budgets`, `resolve_budgets`, `SitePlan`, `Plan`, `make_plan` |
| `src/src_method/_store.py` | create | `EnvironmentStore` |
| `src/src_method/_sites.py` | create | `padded_shapes`, `site_bytes`, `SiteSource` |
| `src/src_method/_sweep.py` | modify, then rewrite | driver over the parts above |
| `src/src_method/stack.py`, `apply.py`, `compress.py` | modify | `resources` keyword |
| `src/src_method/__init__.py` | modify | export `Resources` |
| `tests/test_backend.py`, `tests/test_kernels.py`, `tests/test_plan.py`, `tests/test_store.py`, `tests/test_sites.py` | create | unit tests |
| `tests/test_tensor_train.py`, `tests/test_stack.py`, `tests/test_gpu_backend.py` | modify | padding, integration and GPU tests |
| `benches/large/bench_large.py`, `benches/large/README.md` | create | cluster acceptance benchmark |
| `benches/README.md`, `README.md`, `docs/large-problems.md`, `mkdocs.yml`, `docs/developer-guide/testing.md` | modify / create | documentation |

Test commands pass `-o log_cli=false` only to keep the output short.

---

### Task 1: Backend helpers for streams, staging and memory

**Files:**
- Modify: `src/src_method/utils/_backend.py` (module docstring, imports, append helpers)
- Modify: `src/src_method/utils/__init__.py`
- Test: `tests/test_backend.py` (create)

**Interfaces:**
- Consumes: existing `get_xp`, `default_rng`, `to_numpy`.
- Produces (all exported from `src_method.utils`):
  - `NullEvent` with `synchronize() -> None`; `NullStream` with `record() -> NullEvent`, `wait_event(event) -> None`, `synchronize() -> None`.
  - `is_host(xp) -> bool`; `new_stream(xp)` (non-blocking CuPy stream or `NullStream`); `current_stream(xp)`.
  - `pinned_empty(n_bytes: int, xp) -> np.ndarray` (flat `uint8`, page-locked on CuPy).
  - `to_device_async(host: np.ndarray, xp, stream) -> NDArray` (a copy on the host backend); `to_host_async(device, out: np.ndarray, stream) -> None`.
  - `device_memory(xp) -> tuple[int, int]` (`(available, total)`); `device_pool_bytes(xp) -> int` (0 on the host); `device_pool_limit(xp, budget: int)` context manager; `host_memory_available(meminfo: str = "/proc/meminfo") -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backend.py`:

```python
"""Test the host-side behaviour of the backend helpers."""

import numpy as np

from src_method.utils import (
    NullEvent,
    NullStream,
    current_stream,
    device_pool_bytes,
    device_pool_limit,
    host_memory_available,
    is_host,
    new_stream,
    pinned_empty,
    to_device_async,
    to_host_async,
)


def test_host_streams_are_null():
    stream = new_stream(np)

    assert is_host(np)
    assert isinstance(stream, NullStream)
    assert isinstance(current_stream(np), NullStream)
    event = stream.record()
    assert isinstance(event, NullEvent)
    stream.wait_event(event)
    event.synchronize()
    stream.synchronize()


def test_pinned_empty_on_host_is_a_byte_buffer():
    buffer = pinned_empty(24, np)

    assert buffer.dtype == np.uint8
    assert buffer.shape == (24,)


def test_to_device_async_on_host_copies():
    host = np.arange(6.0).reshape(2, 3)

    device = to_device_async(host, np, NullStream())
    host[:] = 0

    np.testing.assert_array_equal(device, np.arange(6.0).reshape(2, 3))


def test_to_host_async_on_host_fills_out():
    out = np.empty((2, 3))

    to_host_async(np.arange(6.0).reshape(2, 3).T.T, out, NullStream())

    np.testing.assert_array_equal(out, np.arange(6.0).reshape(2, 3))


def test_device_pool_limit_is_a_no_op_on_host():
    with device_pool_limit(np, 10):
        pass


def test_device_pool_is_empty_on_host():
    assert device_pool_bytes(np) == 0


def test_host_memory_reads_mem_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable:    2048 kB\n")

    assert host_memory_available(str(meminfo)) == 2048 * 1024


def test_host_memory_falls_back_without_meminfo(tmp_path):
    assert host_memory_available(str(tmp_path / "missing")) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_backend.py -o log_cli=false`
Expected: FAIL with `ImportError: cannot import name 'NullEvent' from 'src_method.utils'`.

- [ ] **Step 3: Implement the helpers**

Save as `/tmp/t1_backend.diff` and apply from the repository root with `git apply /tmp/t1_backend.diff` (it only touches the working tree):

```diff
--- a/src/src_method/utils/_backend.py
+++ b/src/src_method/utils/_backend.py
@@ -1,18 +1,24 @@
 """Array-module backend selection for CPU (numpy) and GPU (cupy).

-Kept intentionally minimal: a single resolver returns the appropriate
-array module, a PRNG factory, and a host-transfer helper.  All hot-loop
-code paths receive an ``xp`` module and call ``xp.linalg.*`` /
-``xp.asarray`` directly, so backend selection adds zero per-op overhead.
+A single resolver returns the appropriate array module, next to a PRNG factory,
+host-transfer helpers and the few stream and memory queries the out-of-core sweep
+needs. All hot-loop code paths receive an ``xp`` module and call ``xp.linalg.*`` /
+``xp.asarray`` directly, so backend selection adds zero per-op overhead. On the
+host backend the stream helpers are synchronous no-ops, so the same algorithm code
+runs on both.
 """

 from __future__ import annotations

-from typing import TYPE_CHECKING
+import os
+from contextlib import contextmanager
+from pathlib import Path
+from typing import TYPE_CHECKING, Any

 import numpy as np

 if TYPE_CHECKING:
+    from collections.abc import Iterator
     from types import ModuleType

     from numpy.typing import NDArray
@@ -60,3 +66,170 @@
     # cupy.ndarray exposes .get(); fall back to np.asarray for other dispatchers.
     get = getattr(arr, "get", None)
     return get() if callable(get) else np.asarray(arr)
+
+
+class NullEvent:
+    """Stand-in for a CUDA event on the host backend: already complete."""
+
+    def synchronize(self) -> None:
+        """Return at once: host work is synchronous."""
+
+
+class NullStream:
+    """Stand-in for a CUDA stream on the host backend: work runs synchronously."""
+
+    def record(self) -> NullEvent:
+        """Return an event that is already complete."""
+        return NullEvent()
+
+    def wait_event(self, event: NullEvent) -> None:
+        """Return at once: there is nothing to wait for."""
+
+    def synchronize(self) -> None:
+        """Return at once: host work is synchronous."""
+
+
+def is_host(xp: ModuleType) -> bool:
+    """Whether ``xp`` is the host backend (numpy)."""
+    return xp is np
+
+
+def new_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
+    """Return a non-blocking stream for transfers, or a `NullStream` on the host."""
+    if is_host(xp):
+        return NullStream()
+    return xp.cuda.Stream(non_blocking=True)
+
+
+def current_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
+    """Return the stream kernels run on, or a `NullStream` on the host."""
+    if is_host(xp):
+        return NullStream()
+    return xp.cuda.get_current_stream()
+
+
+def pinned_empty(n_bytes: int, xp: ModuleType) -> np.ndarray:
+    """Allocate a flat byte buffer, page-locked on the GPU backend.
+
+    Args:
+        n_bytes: The size of the buffer.
+        xp: Array module (``numpy`` or ``cupy``).
+
+    Returns:
+        A ``uint8`` host array of ``n_bytes`` elements.
+    """
+    if is_host(xp):
+        return np.empty(n_bytes, dtype=np.uint8)
+    import cupyx  # noqa: PLC0415  (lazy: optional dependency)
+
+    return cupyx.empty_pinned(n_bytes, dtype=np.uint8)
+
+
+def to_device_async(host: np.ndarray, xp: ModuleType, stream: Any) -> NDArray:  # noqa: ANN401
+    """Copy a host array to a new device array on ``stream``.
+
+    On the host backend this is a plain copy, so the result never aliases a staging
+    buffer that is about to be reused.
+
+    Args:
+        host: The source; page-locked for the copy to be asynchronous.
+        xp: Array module (``numpy`` or ``cupy``).
+        stream: The stream that performs the copy.
+
+    Returns:
+        The new device array.
+    """
+    if is_host(xp):
+        return host.copy()
+    device = xp.empty(host.shape, dtype=host.dtype)
+    device.set(host, stream=stream)
+    return device
+
+
+def to_host_async(device: NDArray, out: np.ndarray, stream: Any) -> None:  # noqa: ANN401
+    """Copy a device array into a host array on ``stream``.
+
+    The copy is complete once an event recorded on ``stream`` afterwards is.
+
+    Args:
+        device: The source.
+        out: The destination, of the same shape and dtype; page-locked for the
+            copy to be asynchronous.
+        stream: The stream that performs the copy.
+    """
+    if isinstance(device, np.ndarray):
+        np.copyto(out, device)
+        return
+    device.get(stream=stream, out=out, blocking=False)
+
+
+def device_memory(xp: ModuleType) -> tuple[int, int]:
+    """Return the device bytes available to the sweep and the device total.
+
+    Available bytes are the free device memory plus the bytes cached, but unused,
+    by cupy's default memory pool.
+
+    Args:
+        xp: The cupy module.
+
+    Returns:
+        ``(available, total)`` in bytes.
+    """
+    free, total = xp.cuda.runtime.memGetInfo()
+    return free + xp.get_default_memory_pool().free_bytes(), total
+
+
+def device_pool_bytes(xp: ModuleType) -> int:
+    """Return the bytes held by cupy's default memory pool, or 0 on the host.
+
+    The pool keeps freed blocks for reuse, so after a sweep this is its high-water
+    mark.
+    """
+    if is_host(xp):
+        return 0
+    return xp.get_default_memory_pool().total_bytes()
+
+
+@contextmanager
+def device_pool_limit(xp: ModuleType, budget: int) -> Iterator[None]:
+    """Cap cupy's default memory pool at ``budget`` bytes beyond its current use.
+
+    An allocation past the cap fails at once instead of when another allocation
+    runs out. The previous limit is restored on exit. No-op on the host backend.
+
+    Args:
+        xp: Array module (``numpy`` or ``cupy``).
+        budget: The bytes the pool may allocate on top of those in use.
+
+    Yields:
+        Nothing.
+    """
+    if is_host(xp):
+        yield
+        return
+    pool = xp.get_default_memory_pool()
+    previous = pool.get_limit()
+    pool.set_limit(size=pool.used_bytes() + budget)
+    try:
+        yield
+    finally:
+        pool.set_limit(size=previous)
+
+
+def host_memory_available(meminfo: str = "/proc/meminfo") -> int:
+    """Return the host memory available for new allocations, in bytes.
+
+    Reads ``MemAvailable`` on Linux and falls back to the free physical pages.
+
+    Args:
+        meminfo: The path of the ``meminfo`` file.
+
+    Returns:
+        The available bytes.
+    """
+    path = Path(meminfo)
+    if path.exists():
+        for line in path.read_text().splitlines():
+            if line.startswith("MemAvailable:"):
+                return int(line.split()[1]) * 1024
+    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
```

Replace `src/src_method/utils/__init__.py` with:

```python
"""Utility functions for the different SRC algorithms."""

from __future__ import annotations

from ._backend import (
    NullEvent,
    NullStream,
    current_stream,
    default_rng,
    device_memory,
    device_pool_bytes,
    device_pool_limit,
    get_xp,
    host_memory_available,
    is_host,
    new_stream,
    pinned_empty,
    to_device_async,
    to_host_async,
    to_numpy,
)
from .linalg import truncated_qr
from .logging_config import setup_logging

__all__ = [
    "NullEvent",
    "NullStream",
    "current_stream",
    "default_rng",
    "device_memory",
    "device_pool_bytes",
    "device_pool_limit",
    "get_xp",
    "host_memory_available",
    "is_host",
    "new_stream",
    "pinned_empty",
    "setup_logging",
    "to_device_async",
    "to_host_async",
    "to_numpy",
    "truncated_qr",
]
```

`to_host_async` uses `ndarray.get(stream=..., out=..., blocking=False)`, available in CuPy 13 (the `gpu-nvidia` extra requires `cupy-cuda12x>=13`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_backend.py -o log_cli=false`
Expected: 8 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff format src/src_method/utils tests/test_backend.py
uv run ruff check src/src_method/utils tests/test_backend.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/utils/_backend.py, src/src_method/utils/__init__.py, tests/test_backend.py
but commit -b feat/src-out-of-core -m $'feat(utils): ✨ stream, staging and memory helpers for both backends\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/utils/_backend.py`, `src/src_method/utils/__init__.py`, `tests/test_backend.py`; never the untracked `.codegraph/`.

---

### Task 2: Per-site padding and lazily read sites

**Files:**
- Modify: `src/src_method/_tensor_train.py` (imports, `__all__`, `pad`, `transpose_mpo`)
- Test: `tests/test_tensor_train.py` (imports; append two tests and a helper)

**Interfaces:**
- Consumes: `infer_kind`, `TrainKind`.
- Produces:
  - `padded_shape(shape: tuple[int, ...], kind: TrainKind, i: int, last: int) -> tuple[int, int, int, int]`
  - `pad_site(site: NDArray, kind: TrainKind, i: int, last: int) -> NDArray` (views); `pad(train)` is now a loop over it.
  - `SwappedLegs(site)`: `.shape`, `.dtype`, `.ndim` at once, data swapped on `np.asarray`; `transpose_mpo` returns it for sites without `swapaxes`.

- [ ] **Step 1: Write the failing tests**

Save as `/tmp/t2_test_tt.diff` and apply from the repository root with `git apply /tmp/t2_test_tt.diff` (it only touches the working tree):

```diff
--- a/tests/test_tensor_train.py
+++ b/tests/test_tensor_train.py
@@ -4,9 +4,12 @@
 import pytest

 from src_method._tensor_train import (
+    SwappedLegs,
     exact_stack,
     normalize_stack,
     pad,
+    pad_site,
+    padded_shape,
     transpose_mpo,
     unpad,
 )
@@ -174,3 +177,45 @@

     want = dense_two_site(A) @ dense_two_site(B) @ dense_two_site(last)
     np.testing.assert_allclose(dense_two_site(out), want, atol=1e-10)
+
+
+# -------------------------------
+# --- padded_shape / pad_site ---
+# -------------------------------
+
+
+@pytest.mark.parametrize("kind", ["mps", "mpo"])
+@pytest.mark.parametrize("n_sites", [2, 3, 4])
+def test_padded_shape_matches_pad(kind, n_sites, rng):
+    train = (
+        random_mps(n_sites, 3, rng) if kind == "mps" else random_mpo(n_sites, 3, rng)
+    )
+    last = n_sites - 1
+
+    shapes = [padded_shape(t.shape, kind, i, last) for i, t in enumerate(train)]
+
+    assert shapes == [t.shape for t in pad(train)]
+    for i, site in enumerate(train):
+        np.testing.assert_array_equal(pad_site(site, kind, i, last), pad(train)[i])
+
+
+class LazySite:
+    """A site with shape and dtype but no array methods, read by ``np.asarray``."""
+
+    def __init__(self, data):
+        self.data = data
+        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim
+
+    def __array__(self, dtype=None, copy=None):
+        return np.asarray(self.data, dtype=dtype)
+
+
+def test_transpose_mpo_of_lazy_sites(rng):
+    train = random_mpo(3, 2, rng, up=2, down=3)
+
+    transposed = transpose_mpo([LazySite(site) for site in train])
+
+    assert all(isinstance(site, SwappedLegs) for site in transposed)
+    assert [t.shape for t in transposed] == [(2, 3, 2), (2, 2, 3, 2), (2, 3, 2)]
+    np.testing.assert_array_equal(np.asarray(transposed[1]), train[1].swapaxes(-2, -1))
+    assert np.asarray(transposed[0], dtype=np.complex64).dtype == np.complex64
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tensor_train.py -o log_cli=false`
Expected: FAIL with `ImportError: cannot import name 'SwappedLegs'`.

- [ ] **Step 3: Implement**

Save as `/tmp/t2_tt.diff` and apply from the repository root with `git apply /tmp/t2_tt.diff` (it only touches the working tree):

```diff
--- a/src/src_method/_tensor_train.py
+++ b/src/src_method/_tensor_train.py
@@ -16,7 +16,7 @@

 from __future__ import annotations

-from typing import TYPE_CHECKING, Literal
+from typing import TYPE_CHECKING, Any, Literal

 import numpy as np
 from opt_einsum import contract
@@ -42,6 +42,7 @@

 __all__ = [
     "MIN_SRC_SITES",
+    "SwappedLegs",
     "TrainKind",
     "check_exact_supported",
     "exact_compress",
@@ -49,6 +50,8 @@
     "infer_kind",
     "normalize_stack",
     "pad",
+    "pad_site",
+    "padded_shape",
     "transpose_mpo",
     "unpad",
 ]
@@ -137,6 +140,48 @@
     return U[:, :rank] * S[:rank], Vh[:rank]


+def padded_shape(
+    shape: tuple[int, ...], kind: TrainKind, i: int, last: int
+) -> tuple[int, int, int, int]:
+    """Return the bulk ``(l, r, u, d)`` shape that `pad_site` gives a site.
+
+    Args:
+        shape: The unpadded shape of site ``i``.
+        kind: The kind of the train the site belongs to.
+        i: The position of the site.
+        last: The position of the last site of the train.
+
+    Returns:
+        The padded shape.
+    """
+    padded = (*shape, 1) if kind == "mps" else tuple(shape)
+    if i == 0:
+        padded = (1, *padded)
+    if i == last:
+        padded = (padded[0], 1, *padded[1:])
+    return padded  # type: ignore[return-value]
+
+
+def pad_site(site: NDArray, kind: TrainKind, i: int, last: int) -> NDArray:
+    """View one site as a bulk MPO tensor ``(l, r, u, d)``; see `pad`.
+
+    Args:
+        site: The site tensor at position ``i``.
+        kind: The kind of the train the site belongs to.
+        i: The position of the site.
+        last: The position of the last site of the train.
+
+    Returns:
+        The rank-4 view.
+    """
+    view = site[..., None] if kind == "mps" else site
+    if i == 0:
+        view = view[None]
+    if i == last:
+        view = view[:, None]
+    return view
+
+
 def pad(train: Sequence[NDArray]) -> list[NDArray]:
     """View every site of a train as a bulk MPO tensor ``(l, r, u, d)``.

@@ -152,15 +197,7 @@
     """
     kind = infer_kind(train)
     last = len(train) - 1
-    padded = []
-    for i, site in enumerate(train):
-        view = site[..., None] if kind == "mps" else site
-        if i == 0:
-            view = view[None]
-        if i == last:
-            view = view[:, None]
-        padded.append(view)
-    return padded
+    return [pad_site(site, kind, i, last) for i, site in enumerate(train)]


 def unpad(train: Sequence[NDArray], kind: TrainKind) -> list[NDArray]:
@@ -185,9 +222,42 @@
     return unpadded


+class SwappedLegs:
+    """A site read lazily, with its ``u`` and ``d`` legs swapped on reading.
+
+    Stands in for ``site.swapaxes(-2, -1)`` when ``site`` is a lazily loaded
+    array-like (a zarr or HDF5 dataset) that has no ``swapaxes``: the shape is known
+    at once and the data is only read by ``np.asarray``.
+    """
+
+    def __init__(self, site: Any) -> None:  # noqa: ANN401  (any lazy array-like)
+        """Wrap a lazily loaded site.
+
+        Args:
+            site: An array-like with ``shape``, ``dtype`` and ``np.asarray`` support.
+        """
+        self._site = site
+        shape = tuple(site.shape)
+        self.shape = (*shape[:-2], shape[-1], shape[-2])
+        self.dtype = np.dtype(site.dtype)
+        self.ndim = len(shape)
+
+    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:  # noqa: ANN401, FBT001
+        """Read the site and swap its physical legs."""
+        del copy
+        swapped = np.asarray(self._site).swapaxes(-2, -1)
+        return swapped if dtype is None else swapped.astype(dtype)
+
+
 def transpose_mpo(train: Sequence[NDArray]) -> list[NDArray]:
-    """Transpose an MPO by swapping its ``u`` and ``d`` legs on every site (views)."""
-    return [site.swapaxes(-2, -1) for site in train]
+    """Transpose an MPO by swapping its ``u`` and ``d`` legs on every site.
+
+    Arrays give views; lazily loaded sites without ``swapaxes`` give `SwappedLegs`.
+    """
+    return [
+        site.swapaxes(-2, -1) if hasattr(site, "swapaxes") else SwappedLegs(site)
+        for site in train
+    ]


 def normalize_stack(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tensor_train.py tests/test_stack.py -o log_cli=false -m "not perf"`
Expected: all pass (24 in `test_tensor_train.py`).

- [ ] **Step 5: Lint**

```bash
uv run ruff format src/src_method/_tensor_train.py tests/test_tensor_train.py
uv run ruff check src/src_method/_tensor_train.py tests/test_tensor_train.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_tensor_train.py, tests/test_tensor_train.py
but commit -b feat/src-out-of-core -m $'feat(tensor-train): ✨ per-site padding and lazily read sites\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_tensor_train.py`, `tests/test_tensor_train.py`; never the untracked `.codegraph/`.

---

### Task 3: Batched site kernels and their peak memory

**Files:**
- Create: `src/src_method/_kernels.py`
- Modify: `src/src_method/_sweep.py` (drop the equations and the contraction cache, use `SiteKernels`)
- Test: `tests/test_kernels.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Equations(ltr, rtl_m, rtl_s, first)`, `equations(depth: int) -> Equations` (cached; same strings as today's `_equations`).
  - `SiteKernels(depth)` with `.eqs` and `env(env, omega, cores)`, `sketch(env, cores, proj)`, `project(eta, cores, proj)`, `first(cores, proj)`; `cores` is the tuple of padded cores of one site.
  - `peak_elements(eq: str, shapes: tuple[tuple[int, ...], ...]) -> int` (cached): peak elements beyond the inputs, final output included.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kernels.py`:

```python
"""Test the batched site kernels and the peak-memory walk."""

import numpy as np
import pytest

from src_method._kernels import SiteKernels, equations, peak_elements


def cores(rng, dtype, *, left=(3, 2), right=(4, 3), up=2, mid=3, down=2):
    """Two padded layers: (l, r, u, x) and (l, r, x, d)."""

    def draw(*shape):
        out = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            out = out + 1j * rng.normal(size=shape)
        return out.astype(dtype)

    return (
        draw(left[0], right[0], up, mid),
        draw(left[1], right[1], mid, down),
    )


def batched(fn, n, batch, axis):
    parts = [fn(lo, min(lo + batch, n)) for lo in range(0, n, batch)]
    return np.concatenate(parts, axis=axis)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("batch", [1, 3, 7])
def test_batched_kernels_match_unbatched(dtype, batch):
    rng = np.random.default_rng(0)
    k = SiteKernels(2)
    site = cores(rng, dtype)
    chi, eta = 7, 5
    env = rng.normal(size=(chi, 3, 2)).astype(dtype)
    omega = rng.normal(size=(chi, 2, 2)).astype(dtype)
    proj = rng.normal(size=(eta, 4, 3)).astype(dtype)
    out_core = rng.normal(size=(chi, eta, 2, 2)).astype(dtype)

    full_env = k.env(env, omega, site)
    full_sketch = k.sketch(env, site, proj)
    full_proj = k.project(out_core, site, proj)
    full_first = k.first(site, proj)

    np.testing.assert_allclose(
        batched(lambda lo, hi: k.env(env[lo:hi], omega[lo:hi], site), chi, batch, 0),
        full_env,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.sketch(env[lo:hi], site, proj), chi, batch, 3),
        full_sketch,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.project(out_core[lo:hi], site, proj), chi, batch, 0),
        full_proj,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.first(site, proj[lo:hi]), eta, batch, 2),
        full_first,
        rtol=1e-12,
    )


def test_equations_depth_one():
    eqs = equations(1)

    assert eqs.ltr == "ad,afg,defg->ae"
    assert eqs.first == "defg,be->dbfg"


def test_peak_elements_of_a_matrix_chain():
    # (2x3)(3x4)(4x5): the path contracts the first pair into a 2x4 intermediate,
    # holding its two operands (6 + 12) and the output (8), then the last pair,
    # holding that intermediate twice (as live and as operand), the 4x5 operand
    # and the 2x5 output: 8 + 8 + 20 + 10 = 46.
    shapes = ((2, 3), (3, 4), (4, 5))

    assert peak_elements("ab,bc,cd->ad", shapes) == 46
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_kernels.py -o log_cli=false`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_method._kernels'`.

- [ ] **Step 3: Create the kernels module**

Create `src/src_method/_kernels.py`:

```python
"""Site kernels of the SRC sweep and the memory they need.

Every kernel is one of the contractions of the sweep restricted to a batch: a slice
of the sketch index for the environments and the sketch, a slice of the rows of
the new projected environment otherwise. Batching never changes the result beyond
rounding, because the sketch index and those rows are free indices of their
contractions.
"""

from __future__ import annotations

from functools import cache
from itertools import count
from math import prod
from typing import TYPE_CHECKING, NamedTuple

import opt_einsum as oe
from opt_einsum import contract_expression, get_symbol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from opt_einsum.contract import ContractExpression

Shape = tuple[int, ...]


class Equations(NamedTuple):
    """The einsum equations of one sweep, for a fixed stack depth."""

    ltr: str
    rtl_m: str
    rtl_s: str
    first: str


@cache
def equations(depth: int) -> Equations:
    """Generate the sweep equations for a stack of ``depth`` layers.

    Layer ``i`` at a site carries ``(a_i, b_i, x_i, x_{i+1})``: left and right
    bonds, then its upper and lower physical legs, so that consecutive layers share
    ``x``. The output legs are ``x_0`` (up) and ``x_depth`` (down).
    """
    symbols = map(get_symbol, count())
    sketch, eta_right, eta_left = next(symbols), next(symbols), next(symbols)
    left = "".join(next(symbols) for _ in range(depth))
    right = "".join(next(symbols) for _ in range(depth))
    phys = [next(symbols) for _ in range(depth + 1)]
    up, down = phys[0], phys[-1]
    layers = ",".join(
        f"{left[i]}{right[i]}{phys[i]}{phys[i + 1]}" for i in range(depth)
    )
    return Equations(
        ltr=f"{sketch}{left},{sketch}{up}{down},{layers}->{sketch}{right}",
        rtl_m=f"{sketch}{left},{layers},{eta_right}{right}->{eta_right}{up}{down}{sketch}",
        rtl_s=f"{eta_left}{eta_right}{up}{down},{layers},{eta_right}{right}->{eta_left}{left}",
        first=f"{layers},{eta_right}{right}->{left}{eta_right}{up}{down}",
    )


class _Contractions:
    """Compiled contractions for one sweep, keyed on equation and operand shapes.

    Uniform bulk sites share one entry, so the path is planned once rather than at
    every site. Kept per call: jagged bonds and short last batches add entries that
    are not worth keeping.
    """

    def __init__(self) -> None:
        self._compiled: dict[tuple[str, tuple[Shape, ...]], ContractExpression] = {}

    def __call__(self, eq: str, *operands: NDArray) -> NDArray:
        shapes = tuple(op.shape for op in operands)
        expr = self._compiled.get((eq, shapes))
        if expr is None:
            expr = self._compiled[eq, shapes] = contract_expression(eq, *shapes)
        return expr(*operands)


class SiteKernels:
    """The four contractions of the sweep, each applied to one batch.

    Args:
        depth: The number of layers of the stack.
    """

    def __init__(self, depth: int) -> None:
        self.eqs = equations(depth)
        self._contract = _Contractions()

    def env(self, env: NDArray, omega: NDArray, cores: Sequence[NDArray]) -> NDArray:
        """Advance a batch of sketch columns of the environment by one site.

        Args:
            env: Columns ``lo:hi`` of ``C_j``, shape ``(b, *left_bonds)``.
            omega: The same columns of the site's Gaussian tensor, ``(b, up, down)``.
            cores: The padded cores of the site.

        Returns:
            Columns ``lo:hi`` of ``C_{j+1}``, shape ``(b, *right_bonds)``.
        """
        return self._contract(self.eqs.ltr, env, omega, *cores)

    def sketch(self, env: NDArray, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Sketch a batch of columns of the site's running core.

        Args:
            env: Columns ``lo:hi`` of ``C_j``, shape ``(b, *left_bonds)``.
            cores: The padded cores of the site.
            proj: The projected environment ``S``, ``(eta, *right_bonds)``.

        Returns:
            Columns ``lo:hi`` of the sketch, shape ``(eta, up, down, b)``.
        """
        return self._contract(self.eqs.rtl_m, env, *cores, proj)

    def project(self, eta: NDArray, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Project a batch of rows of the new projected environment.

        Args:
            eta: Rows ``lo:hi`` of the conjugated output core, ``(b, eta, up, down)``.
            cores: The padded cores of the site.
            proj: The projected environment ``S``, ``(eta, *right_bonds)``.

        Returns:
            Rows ``lo:hi`` of the new ``S``, shape ``(b, *left_bonds)``.
        """
        return self._contract(self.eqs.rtl_s, eta, *cores, proj)

    def first(self, cores: Sequence[NDArray], proj: NDArray) -> NDArray:
        """Contract the first site with a batch of rows of ``S``.

        Args:
            cores: The padded cores of the first site.
            proj: Rows ``lo:hi`` of ``S``, ``(b, *right_bonds)``.

        Returns:
            The output core for those rows, ``(*left_bonds, b, up, down)``.
        """
        return self._contract(self.eqs.first, *cores, proj)


@cache
def peak_elements(eq: str, shapes: tuple[Shape, ...]) -> int:
    """Estimate the peak elements a contraction allocates beyond its inputs.

    Walks the path `opt_einsum` picks for these shapes, the one `SiteKernels` runs.
    Each pairwise step holds the intermediates still alive, its output and a
    possible contiguous copy of both operands (``tensordot`` transposes them), so
    the estimate errs on the high side. The final output is included, the inputs
    are not.

    Args:
        eq: The einsum equation.
        shapes: The operand shapes.

    Returns:
        The peak number of elements.
    """
    _, info = oe.contract_path(eq, *shapes, shapes=True)
    sizes = info.size_dict
    # (elements, is_intermediate) for every operand still to be contracted.
    operands = [(prod(shape), False) for shape in shapes]
    peak = 0
    for step in info.contraction_list:
        positions, einsum_str = step[0], step[2]
        live = sum(n for n, is_tmp in operands if is_tmp)
        popped = [operands.pop(i) for i in positions]
        out = prod(sizes[c] for c in einsum_str.split("->")[1])
        peak = max(peak, live + out + sum(n for n, _ in popped))
        operands.append((out, True))
    return peak
```

- [ ] **Step 4: Switch the sweep to the kernels**

Save as `/tmp/t3_sweep.diff` and apply from the repository root with `git apply /tmp/t3_sweep.diff` (it only touches the working tree):

```diff
--- a/src/src_method/_sweep.py
+++ b/src/src_method/_sweep.py
@@ -13,16 +13,14 @@

 from __future__ import annotations

-from functools import cache
-from itertools import count
 from math import prod
 from time import perf_counter_ns
-from typing import TYPE_CHECKING, NamedTuple
+from typing import TYPE_CHECKING

 import numpy as np
 import structlog
-from opt_einsum import contract_expression, get_symbol

+from ._kernels import SiteKernels
 from ._tensor_train import pad, unpad
 from .utils import to_numpy, truncated_qr

@@ -31,67 +29,12 @@
     from types import ModuleType

     from numpy.typing import NDArray
-    from opt_einsum.contract import ContractExpression

     from ._tensor_train import TrainKind

 logger = structlog.get_logger(__name__)


-class _Contractions:
-    """Compiled contractions for one sweep, keyed on equation and operand shapes.
-
-    Uniform bulk sites share one entry, so the path is planned once rather than at
-    every site. Kept per call: jagged bonds add entries that are not worth keeping.
-    """
-
-    def __init__(self) -> None:
-        self._compiled: dict[
-            tuple[str, tuple[tuple[int, ...], ...]], ContractExpression
-        ] = {}
-
-    def __call__(self, eq: str, *operands: NDArray) -> NDArray:
-        shapes = tuple(op.shape for op in operands)
-        expr = self._compiled.get((eq, shapes))
-        if expr is None:
-            expr = self._compiled[eq, shapes] = contract_expression(eq, *shapes)
-        return expr(*operands)
-
-
-class _Equations(NamedTuple):
-    """The einsum equations of one sweep, for a fixed stack depth."""
-
-    ltr: str
-    rtl_m: str
-    rtl_s: str
-    first: str
-
-
-@cache
-def _equations(depth: int) -> _Equations:
-    """Generate the sweep equations for a stack of ``depth`` layers.
-
-    Layer ``i`` at a site carries ``(a_i, b_i, x_i, x_{i+1})``: left and right
-    bonds, then its upper and lower physical legs, so that consecutive layers share
-    ``x``. The output legs are ``x_0`` (up) and ``x_depth`` (down).
-    """
-    symbols = map(get_symbol, count())
-    sketch, eta_right, eta_left = next(symbols), next(symbols), next(symbols)
-    left = "".join(next(symbols) for _ in range(depth))
-    right = "".join(next(symbols) for _ in range(depth))
-    phys = [next(symbols) for _ in range(depth + 1)]
-    up, down = phys[0], phys[-1]
-    layers = ",".join(
-        f"{left[i]}{right[i]}{phys[i]}{phys[i + 1]}" for i in range(depth)
-    )
-    return _Equations(
-        ltr=f"{sketch}{left},{sketch}{up}{down},{layers}->{sketch}{right}",
-        rtl_m=f"{sketch}{left},{layers},{eta_right}{right}->{eta_right}{up}{down}{sketch}",
-        rtl_s=f"{eta_left}{eta_right}{up}{down},{layers},{eta_right}{right}->{eta_left}{left}",
-        first=f"{layers},{eta_right}{right}->{left}{eta_right}{up}{down}",
-    )
-
-
 def sweep(
     layers: Sequence[Sequence[NDArray]],
     kind: TrainKind,
@@ -121,8 +64,7 @@
     """
     depth = len(layers)
     n_sites = len(layers[0])
-    eqs = _equations(depth)
-    contract = _Contractions()
+    kernels = SiteKernels(depth)
     # sites[j] holds the padded tensors of every layer at site j.
     sites = list(zip(*(pad([xp.asarray(a) for a in layer]) for layer in layers)))
     logger.debug(
@@ -137,7 +79,7 @@
     for j in range(n_sites - 1):
         up, down = sites[j][0].shape[2], sites[j][-1].shape[3]
         omega = xp.asarray(prng.normal(size=(chi_out, up, down))).astype(dtype)
-        C.append(contract(eqs.ltr, C[j], omega, *sites[j]))
+        C.append(kernels.env(C[j], omega, sites[j]))
     logger.debug("Left-to-right sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

     tms = perf_counter_ns()
@@ -145,13 +87,13 @@
     S = xp.ones((1,) * (depth + 1), dtype=dtype)
     for j in range(n_sites - 1, 0, -1):
         # C[-1] is C[j] here; popping it frees each environment once used.
-        M = contract(eqs.rtl_m, C.pop(), *sites[j], S)
+        M = kernels.sketch(C.pop(), sites[j], S)
         rows = M.shape[0] * M.shape[1] * M.shape[2]
         Q = truncated_qr(M.reshape(rows, chi_out), cutoff, xp)
         eta_j = Q.reshape(*M.shape[:3], Q.shape[1]).transpose(3, 0, 1, 2)
-        S = contract(eqs.rtl_s, eta_j.conj(), *sites[j], S)
+        S = kernels.project(eta_j.conj(), sites[j], S)
         eta_reversed.append(eta_j)
-    first = contract(eqs.first, *sites[0], S)
+    first = kernels.first(sites[0], S)
     eta = [first.reshape(1, *first.shape[depth:]), *reversed(eta_reversed)]
     logger.debug("Right-to-left sweep", seconds=(perf_counter_ns() - tms) * 1e-9)

```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -o log_cli=false -m "not perf"`
Expected: all pass, including the 8 tests of `tests/test_kernels.py`.

- [ ] **Step 6: Lint**

```bash
uv run ruff format src/src_method/_kernels.py src/src_method/_sweep.py tests/test_kernels.py
uv run ruff check src/src_method/_kernels.py src/src_method/_sweep.py tests/test_kernels.py
```
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_kernels.py, src/src_method/_sweep.py, tests/test_kernels.py
but commit -b feat/src-out-of-core -m $'refactor(sweep): ♻️ batched site kernels with a peak-memory estimate\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_kernels.py`, `src/src_method/_sweep.py`, `tests/test_kernels.py`; never the untracked `.codegraph/`.

---

### Task 4: `Resources` and budget detection

**Files:**
- Create: `src/src_method/_plan.py` (budgets only; the planner follows in Task 5)
- Modify: `src/src_method/__init__.py` (export `Resources`)
- Test: `tests/test_plan.py` (create)

**Interfaces:**
- Consumes: `device_memory`, `host_memory_available`, `is_host` from Task 1.
- Produces:
  - `parse_size(value: int | str) -> int`; `TypeError` for other types, `ValueError` for negative or unrecognised values (messages start with `Expected`).
  - `Resources(gpu_memory=None, host_memory=None, scratch_dir=None)`, frozen, validated at construction; exported as `src_method.Resources`.
  - `Budgets(device: int, host: int, disk: int, scratch_dir: Path, unified: bool)`, frozen.
  - `resolve_budgets(resources: Resources | None, xp) -> Budgets`; module constants `GPU_MARGIN_FRACTION = 0.10`, `GPU_MARGIN_MIN = 2**30`, `HOST_MARGIN_FRACTION = 0.10`, `DISK_MARGIN_FRACTION = 0.05`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan.py`:

```python
"""Test the memory planner: sizes, budgets, batches and tiers."""

import numpy as np
import pytest

import src_method._plan as plan_module
from src_method import Resources
from src_method._plan import parse_size, resolve_budgets

GB = 10**9


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        ("36GB", 36 * 10**9),
        ("36GiB", 36 * 2**30),
        ("1.5 kB", 1500),
        ("512", 512),
        ("2mib", 2 * 2**20),
    ],
)
def test_parse_size(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", [-1, "36 GBs", "lots", ""])
def test_parse_size_rejects_values(value):
    with pytest.raises(ValueError, match="Expected"):
        parse_size(value)


@pytest.mark.parametrize("value", [True, 1.5, None])
def test_parse_size_rejects_types(value):
    with pytest.raises(TypeError, match="Expected"):
        parse_size(value)


def test_resources_validate_at_construction():
    with pytest.raises(ValueError, match="size string"):
        Resources(gpu_memory="plenty")


def test_resolve_budgets_explicit_on_host(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="2GB", scratch_dir=tmp_path), np)

    assert budgets.unified
    assert budgets.host == budgets.device == 2 * GB
    assert budgets.scratch_dir == tmp_path
    assert budgets.disk > 0


def test_resolve_budgets_detects_host_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(plan_module, "host_memory_available", lambda: 10 * GB)

    budgets = resolve_budgets(Resources(scratch_dir=tmp_path / "not" / "yet"), np)

    assert budgets.host == 9 * GB


def test_resolve_budgets_detects_device_memory(monkeypatch, tmp_path):
    fake_xp = object()
    monkeypatch.setattr(plan_module, "is_host", lambda _xp: False)
    monkeypatch.setattr(plan_module, "device_memory", lambda _xp: (30 * GB, 40 * GB))

    budgets = resolve_budgets(
        Resources(host_memory="1GB", scratch_dir=tmp_path), fake_xp
    )

    assert not budgets.unified
    assert budgets.device == 26 * GB  # minus max(10% of 40 GB, 1 GiB)


def test_resolve_budgets_detects_disk(monkeypatch, tmp_path):
    usage = type("Usage", (), {"free": 100 * GB})
    monkeypatch.setattr(plan_module.shutil, "disk_usage", lambda _path: usage)

    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), np)

    assert budgets.disk == 95 * GB
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_plan.py -o log_cli=false`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_method._plan'`.

- [ ] **Step 3: Implement the budgets**

Create `src/src_method/_plan.py`:

```python
"""Memory planning for the SRC sweep: budgets, batch sizes and environment tiers."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import device_memory, host_memory_available, is_host

if TYPE_CHECKING:
    import os
    from types import ModuleType

# Kept free on the device for cuBLAS/cuSOLVER workspaces and pool fragmentation.
GPU_MARGIN_FRACTION = 0.10
GPU_MARGIN_MIN = 2**30
HOST_MARGIN_FRACTION = 0.10
DISK_MARGIN_FRACTION = 0.05

_UNITS = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?\s*$", re.IGNORECASE)


def parse_size(value: int | str) -> int:
    """Convert a byte count or a size string to bytes.

    Args:
        value: A non-negative integer, or a string such as ``"36GB"`` (decimal,
            ``36 * 10**9``) or ``"36GiB"`` (binary, ``36 * 2**30``).

    Returns:
        The size in bytes.

    Raises:
        TypeError: If the value is neither an integer nor a string.
        ValueError: If the value is negative or not a recognised size.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"Expected a byte count or a size string such as '36GB', got {value!r}."
        raise TypeError(msg)
    if isinstance(value, int):
        if value < 0:
            msg = f"Expected a non-negative byte count, got {value}."
            raise ValueError(msg)
        return value
    match = _SIZE.match(value)
    if match is None:
        msg = f"Expected a size string such as '36GB' or '36GiB', got {value!r}."
        raise ValueError(msg)
    number, unit = match.groups()
    return int(float(number) * _UNITS[(unit or "B").upper()])


@dataclass(frozen=True)
class Resources:
    """Memory budgets and scratch space for one `src` call.

    Every field left as ``None`` is detected when the call starts.

    Attributes:
        gpu_memory: Device memory the sweep may use, as a byte count or a size
            string (``"36GB"``, ``"36GiB"``). Defaults to the free device memory
            minus ``max(10%, 1 GiB)``. Ignored on the CPU.
        host_memory: Host memory the sweep may use. Defaults to ``MemAvailable``
            minus 10%. On the CPU it covers the working set as well.
        scratch_dir: Directory for environments that fit in neither budget,
            ideally on node-local disk. Defaults to ``tempfile.gettempdir()``,
            which honours ``$TMPDIR``.
    """

    gpu_memory: int | str | None = None
    host_memory: int | str | None = None
    scratch_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        """Validate the explicit budgets.

        Raises:
            TypeError: If a budget is neither an integer nor a string.
            ValueError: If a budget is not a recognised size.
        """
        for value in (self.gpu_memory, self.host_memory):
            if value is not None:
                parse_size(value)


@dataclass(frozen=True)
class Budgets:
    """The resolved budgets of one sweep, in bytes.

    Attributes:
        device: Bytes for the working set and the device tier.
        host: Bytes for the output, the staging buffers and the host tier.
        disk: Bytes free in ``scratch_dir``.
        scratch_dir: Where the disk tier lives.
        unified: Whether device and host memory are the same (the CPU backend).
    """

    device: int
    host: int
    disk: int
    scratch_dir: Path
    unified: bool


def resolve_budgets(resources: Resources | None, xp: ModuleType) -> Budgets:
    """Turn `Resources` into byte budgets, detecting those left unset.

    Args:
        resources: The requested budgets, or ``None`` to detect all of them.
        xp: Array module (``numpy`` or ``cupy``).

    Returns:
        The budgets.
    """
    resources = resources or Resources()
    scratch = (
        Path(resources.scratch_dir)
        if resources.scratch_dir is not None
        else Path(tempfile.gettempdir())
    )
    if resources.host_memory is not None:
        host = parse_size(resources.host_memory)
    else:
        host = int(host_memory_available() * (1 - HOST_MARGIN_FRACTION))
    unified = is_host(xp)
    if unified:
        device = host
    elif resources.gpu_memory is not None:
        device = parse_size(resources.gpu_memory)
    else:
        available, total = device_memory(xp)
        device = available - max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
    free = shutil.disk_usage(_existing_parent(scratch)).free
    disk = int(free * (1 - DISK_MARGIN_FRACTION))
    return Budgets(max(device, 0), max(host, 0), disk, scratch, unified)


def _existing_parent(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path
```

Save as `/tmp/t4_init.diff` and apply from the repository root with `git apply /tmp/t4_init.diff` (it only touches the working tree):

```diff
--- a/src/src_method/__init__.py
+++ b/src/src_method/__init__.py
@@ -5,10 +5,11 @@

 from __future__ import annotations

+from ._plan import Resources
 from ._version import version as __version__
 from ._version import version_tuple as __version_tuple__
 from .apply import apply
 from .compress import compress
 from .stack import src

-__all__ = ["__version__", "__version_tuple__", "apply", "compress", "src"]
+__all__ = ["Resources", "__version__", "__version_tuple__", "apply", "compress", "src"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plan.py -o log_cli=false`
Expected: 18 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff format src/src_method/_plan.py src/src_method/__init__.py tests/test_plan.py
uv run ruff check src/src_method/_plan.py src/src_method/__init__.py tests/test_plan.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_plan.py, src/src_method/__init__.py, tests/test_plan.py
but commit -b feat/src-out-of-core -m $'feat(plan): ✨ Resources and memory budget detection\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_plan.py`, `src/src_method/__init__.py`, `tests/test_plan.py`; never the untracked `.codegraph/`.

---

### Task 5: The planner: batch sizes, tiers and prefetching

**Files:**
- Modify: `src/src_method/_plan.py` (replace with the full planner)
- Test: `tests/test_plan.py` (replace with the full test file)

**Interfaces:**
- Consumes: `equations`, `peak_elements` (Task 3); Task 4's budgets.
- Produces:
  - `Tier = Literal["device", "host", "disk"]`; `PREFERRED_BATCH = 512`; `GEMM_MULTIPLE = 32`.
  - `SitePlan(env_batch: int, sketch_batch: int, project_batch: int, tier: Tier)`: `env_batch` is 0 at the last site, `sketch_batch` 0 at site 0, `project_batch` at site 0 is the first-site step; `tier` is where `C_j` lives (`"device"` at site 0, which stores none).
  - `Plan(sites: tuple[SitePlan, ...], prefetch: int, device_peak: int, host_peak: int, disk_bytes: int)`.
  - `make_plan(site_shapes, site_bytes, chi_out, dtype, budgets) -> Plan`, raising `MemoryError` with `"Site {j}: the {kernel} step needs {n} bytes with a batch of one, ..."` or `"The environments need {n} bytes on disk, ..."`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_plan.py` with:

```python
"""Test the memory planner: sizes, budgets, batches and tiers."""

from pathlib import Path

import numpy as np
import pytest

import src_method._plan as plan_module
from src_method import Resources
from src_method._plan import (
    GEMM_MULTIPLE,
    Budgets,
    make_plan,
    parse_size,
    resolve_budgets,
)

GB = 10**9


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        ("36GB", 36 * 10**9),
        ("36GiB", 36 * 2**30),
        ("1.5 kB", 1500),
        ("512", 512),
        ("2mib", 2 * 2**20),
    ],
)
def test_parse_size(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", [-1, "36 GBs", "lots", ""])
def test_parse_size_rejects_values(value):
    with pytest.raises(ValueError, match="Expected"):
        parse_size(value)


@pytest.mark.parametrize("value", [True, 1.5, None])
def test_parse_size_rejects_types(value):
    with pytest.raises(TypeError, match="Expected"):
        parse_size(value)


def test_resources_validate_at_construction():
    with pytest.raises(ValueError, match="size string"):
        Resources(gpu_memory="plenty")


def test_resolve_budgets_explicit_on_host(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="2GB", scratch_dir=tmp_path), np)

    assert budgets.unified
    assert budgets.host == budgets.device == 2 * GB
    assert budgets.scratch_dir == tmp_path
    assert budgets.disk > 0


def test_resolve_budgets_detects_host_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(plan_module, "host_memory_available", lambda: 10 * GB)

    budgets = resolve_budgets(Resources(scratch_dir=tmp_path / "not" / "yet"), np)

    assert budgets.host == 9 * GB


def test_resolve_budgets_detects_device_memory(monkeypatch, tmp_path):
    fake_xp = object()
    monkeypatch.setattr(plan_module, "is_host", lambda _xp: False)
    monkeypatch.setattr(plan_module, "device_memory", lambda _xp: (30 * GB, 40 * GB))

    budgets = resolve_budgets(
        Resources(host_memory="1GB", scratch_dir=tmp_path), fake_xp
    )

    assert not budgets.unified
    assert budgets.device == 26 * GB  # minus max(10% of 40 GB, 1 GiB)


def test_resolve_budgets_detects_disk(monkeypatch, tmp_path):
    usage = type("Usage", (), {"free": 100 * GB})
    monkeypatch.setattr(plan_module.shutil, "disk_usage", lambda _path: usage)

    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), np)

    assert budgets.disk == 95 * GB


def mpo_stack_shapes(n_sites, bonds, phys=4):
    """Padded shapes of a stack of MPOs, one bond dimension per layer."""

    def shape(j, bond):
        return (1 if j == 0 else bond, 1 if j == n_sites - 1 else bond, phys, phys)

    return [tuple(shape(j, bond) for bond in bonds) for j in range(n_sites)]


def site_bytes(shapes, itemsize=16):
    return [sum(int(np.prod(s)) * itemsize for s in site) for site in shapes]


def budgets(device, host=None, disk=10**15, *, unified=False):
    return Budgets(
        device, device if host is None else host, disk, Path("/scratch"), unified
    )


def test_small_problem_is_one_batch_on_the_device():
    shapes = mpo_stack_shapes(6, [2, 3, 2])

    plan = make_plan(shapes, site_bytes(shapes), 64, np.complex128, budgets(GB))

    assert plan.prefetch == 1
    assert all(site.tier == "device" for site in plan.sites)
    assert plan.sites[0].env_batch == 64
    assert plan.sites[-1].env_batch == 0
    assert plan.sites[0].sketch_batch == 0
    assert all(site.sketch_batch == 64 for site in plan.sites[1:])
    assert all(site.project_batch == 64 for site in plan.sites)
    assert plan.disk_bytes == 0


def test_tight_budget_shrinks_batches_to_gemm_multiples():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])
    chi = 512
    loose = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))

    tight = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(loose.device_peak // 4)
    )

    batches = [s.sketch_batch for s in tight.sites[1:]]
    assert min(batches) < chi
    assert all(b % GEMM_MULTIPLE == 0 for b in batches if b >= GEMM_MULTIPLE)
    assert tight.device_peak <= loose.device_peak // 4


def test_tiers_go_newest_first():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    env = chi * 4 * 4 * 64 * 4 * 16  # one bulk environment, complex128
    roomy = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))

    seen = set()
    for extra in range(8):
        plan = make_plan(
            shapes,
            site_bytes(shapes),
            chi,
            np.complex128,
            budgets(roomy.device_peak - extra * env, host=roomy.host_peak + 7 * env),
        )
        tiers = [site.tier for site in plan.sites[1:]]
        # Oldest sites on the slowest tier: disk, then host, then device.
        assert tiers == sorted(tiers, key=["disk", "host", "device"].index)
        assert plan.disk_bytes == env * tiers.count("disk")
        seen.update(tiers)
    assert seen == {"device", "host", "disk"}


def test_unified_memory_has_no_host_tier():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    roomy = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB, unified=True)
    )

    plan = make_plan(
        shapes,
        site_bytes(shapes),
        chi,
        np.complex128,
        budgets(roomy.device_peak // 2, unified=True),
    )

    tiers = {site.tier for site in plan.sites[1:]}
    assert "host" not in tiers
    assert "disk" in tiers


def test_infeasible_site_names_the_site():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])

    with pytest.raises(MemoryError, match=r"Site \d+: the \w+ step needs \d+ bytes"):
        make_plan(shapes, site_bytes(shapes), 256, np.complex128, budgets(10**6))


def test_environments_must_fit_on_disk():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    roomy = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))
    work = roomy.device_peak - 7 * chi * 4 * 4 * 64 * 4 * 16

    with pytest.raises(MemoryError, match="bytes on disk"):
        make_plan(
            shapes,
            site_bytes(shapes),
            chi,
            np.complex128,
            budgets(work, host=roomy.host_peak, disk=1000),
        )


def test_prefetch_is_dropped_before_giving_up():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])
    chi = 64
    cores = max(site_bytes(shapes))
    with_prefetch = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB)
    )
    # Remove about one site of cores from the smallest budget that fits one column.
    minimum = with_prefetch.device_peak
    lo, hi = 0, minimum
    while lo < hi:  # the smallest budget that plans with prefetching
        mid = (lo + hi) // 2
        try:
            make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(mid))
            hi = mid
        except MemoryError:
            lo = mid + 1

    plan = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(lo))
    assert plan.prefetch in {0, 1}
    if plan.prefetch == 1:
        dropped = make_plan(
            shapes, site_bytes(shapes), chi, np.complex128, budgets(lo - cores // 2)
        )
        assert dropped.prefetch == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plan.py -o log_cli=false`
Expected: FAIL with `ImportError: cannot import name 'GEMM_MULTIPLE'`.

- [ ] **Step 3: Implement the planner**

Replace `src/src_method/_plan.py` with:

```python
"""Memory planning for the SRC sweep: budgets, batch sizes and environment tiers.

`make_plan` is a pure function of the core shapes, the sketch size, the dtype and
the budgets, so the plan of a run can be inspected, and tested, without loading any
data or touching a GPU.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._kernels import equations, peak_elements
from .utils import device_memory, host_memory_available, is_host

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Sequence
    from types import ModuleType

Tier = Literal["device", "host", "disk"]
Shape = tuple[int, ...]

# Kept free on the device for cuBLAS/cuSOLVER workspaces and pool fragmentation.
GPU_MARGIN_FRACTION = 0.10
GPU_MARGIN_MIN = 2**30
HOST_MARGIN_FRACTION = 0.10
DISK_MARGIN_FRACTION = 0.05
# Batch size used to decide how much of the device can hold environments.
PREFERRED_BATCH = 512
# Batches are rounded down to a multiple of this for efficient GEMMs.
GEMM_MULTIPLE = 32

_UNITS = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?\s*$", re.IGNORECASE)


def parse_size(value: int | str) -> int:
    """Convert a byte count or a size string to bytes.

    Args:
        value: A non-negative integer, or a string such as ``"36GB"`` (decimal,
            ``36 * 10**9``) or ``"36GiB"`` (binary, ``36 * 2**30``).

    Returns:
        The size in bytes.

    Raises:
        TypeError: If the value is neither an integer nor a string.
        ValueError: If the value is negative or not a recognised size.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"Expected a byte count or a size string such as '36GB', got {value!r}."
        raise TypeError(msg)
    if isinstance(value, int):
        if value < 0:
            msg = f"Expected a non-negative byte count, got {value}."
            raise ValueError(msg)
        return value
    match = _SIZE.match(value)
    if match is None:
        msg = f"Expected a size string such as '36GB' or '36GiB', got {value!r}."
        raise ValueError(msg)
    number, unit = match.groups()
    return int(float(number) * _UNITS[(unit or "B").upper()])


@dataclass(frozen=True)
class Resources:
    """Memory budgets and scratch space for one `src` call.

    Every field left as ``None`` is detected when the call starts.

    Attributes:
        gpu_memory: Device memory the sweep may use, as a byte count or a size
            string (``"36GB"``, ``"36GiB"``). Defaults to the free device memory
            minus ``max(10%, 1 GiB)``. Ignored on the CPU.
        host_memory: Host memory the sweep may use. Defaults to ``MemAvailable``
            minus 10%. On the CPU it covers the working set as well.
        scratch_dir: Directory for environments that fit in neither budget,
            ideally on node-local disk. Defaults to ``tempfile.gettempdir()``,
            which honours ``$TMPDIR``.
    """

    gpu_memory: int | str | None = None
    host_memory: int | str | None = None
    scratch_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        """Validate the explicit budgets.

        Raises:
            TypeError: If a budget is neither an integer nor a string.
            ValueError: If a budget is not a recognised size.
        """
        for value in (self.gpu_memory, self.host_memory):
            if value is not None:
                parse_size(value)


@dataclass(frozen=True)
class Budgets:
    """The resolved budgets of one sweep, in bytes.

    Attributes:
        device: Bytes for the working set and the device tier.
        host: Bytes for the output, the staging buffers and the host tier.
        disk: Bytes free in ``scratch_dir``.
        scratch_dir: Where the disk tier lives.
        unified: Whether device and host memory are the same (the CPU backend).
    """

    device: int
    host: int
    disk: int
    scratch_dir: Path
    unified: bool


def resolve_budgets(resources: Resources | None, xp: ModuleType) -> Budgets:
    """Turn `Resources` into byte budgets, detecting those left unset.

    Args:
        resources: The requested budgets, or ``None`` to detect all of them.
        xp: Array module (``numpy`` or ``cupy``).

    Returns:
        The budgets.
    """
    resources = resources or Resources()
    scratch = (
        Path(resources.scratch_dir)
        if resources.scratch_dir is not None
        else Path(tempfile.gettempdir())
    )
    if resources.host_memory is not None:
        host = parse_size(resources.host_memory)
    else:
        host = int(host_memory_available() * (1 - HOST_MARGIN_FRACTION))
    unified = is_host(xp)
    if unified:
        device = host
    elif resources.gpu_memory is not None:
        device = parse_size(resources.gpu_memory)
    else:
        available, total = device_memory(xp)
        device = available - max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
    free = shutil.disk_usage(_existing_parent(scratch)).free
    disk = int(free * (1 - DISK_MARGIN_FRACTION))
    return Budgets(max(device, 0), max(host, 0), disk, scratch, unified)


def _existing_parent(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path


@dataclass(frozen=True)
class SitePlan:
    """How one site is processed.

    Attributes:
        env_batch: Sketch columns per environment step (0 at the last site).
        sketch_batch: Sketch columns per sketch step (0 at the first site).
        project_batch: Rows per projection step, or per first-site step at site 0.
        tier: Where ``C_j`` is kept; site 0 stores no environment.
    """

    env_batch: int
    sketch_batch: int
    project_batch: int
    tier: Tier


@dataclass(frozen=True)
class Plan:
    """The memory plan of one sweep.

    Attributes:
        sites: One entry per site.
        prefetch: How many sites ahead the cores are loaded (0 or 1).
        device_peak: Estimated peak device bytes.
        host_peak: Estimated peak host bytes, beyond the inputs.
        disk_bytes: Bytes spilled to the scratch directory.
    """

    sites: tuple[SitePlan, ...]
    prefetch: int
    device_peak: int
    host_peak: int
    disk_bytes: int


class _Site:
    """The memory model of one site, in bytes."""

    def __init__(
        self,
        j: int,
        shapes: tuple[Shape, ...],
        core_bytes: int,
        *,
        chi: int,
        itemsize: int,
        n_sites: int,
    ) -> None:
        self.j, self.shapes, self.core_bytes, self.chi, self.e = (
            j,
            shapes,
            core_bytes,
            chi,
            itemsize,
        )
        self.eqs = equations(len(shapes))
        self.left = tuple(s[0] for s in shapes)
        self.right = tuple(s[1] for s in shapes)
        self.a, self.b = prod(self.left), prod(self.right)
        self.up, self.down = shapes[0][2], shapes[-1][3]
        self.p = self.up * self.down
        # Rows of the projected environment S that enters site j right-to-left.
        self.eta = 1 if j == n_sites - 1 else chi
        self.env_bytes = chi * self.a * itemsize

    def env(self, b: int, prefetch: int, *, staged_in: bool, staged_out: bool) -> int:
        """Bytes of one environment step on ``b`` columns."""
        e, chi = self.e, self.chi
        fixed = self.core_bytes * (1 + prefetch) + chi * self.p * e
        slices = (1 + staged_in) * b * self.a * e + staged_out * b * self.b * e
        peak = peak_elements(
            self.eqs.ltr, ((b, *self.left), (b, self.up, self.down), *self.shapes)
        )
        return fixed + slices + peak * e

    def sketch(self, b: int, prefetch: int, *, staged: bool) -> int:
        """Bytes of one sketch step on ``b`` columns."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.eta * self.p * chi * e
        )
        slices = (1 + staged) * b * self.a * e
        peak = (
            peak_elements(
                self.eqs.rtl_m,
                ((b, *self.left), *self.shapes, (self.eta, *self.right)),
            )
            * e
        )
        return fixed + slices + peak

    def qr(self, prefetch: int) -> int:
        """Bytes of the QR of the sketch: the sketch, ``Q`` and a workspace."""
        e = self.e
        return (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + 3 * self.eta * self.p * self.chi * e
        )

    def project(self, b: int, prefetch: int) -> int:
        """Bytes of one projection step on ``b`` rows."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.eta * self.p * chi * e
            + chi * self.a * e
        )
        slices = b * self.eta * self.p * e
        peak = (
            peak_elements(
                self.eqs.rtl_s,
                (
                    (b, self.eta, self.up, self.down),
                    *self.shapes,
                    (self.eta, *self.right),
                ),
            )
            * e
        )
        return fixed + slices + peak

    def first(self, b: int, prefetch: int) -> int:
        """Bytes of one first-site step on ``b`` rows of ``S``."""
        e, chi = self.e, self.chi
        fixed = (
            self.core_bytes * (1 + prefetch)
            + self.eta * self.b * e
            + self.a * chi * self.p * e
        )
        peak = peak_elements(self.eqs.first, (*self.shapes, (b, *self.right))) * e
        return fixed + peak


def _largest_batch(cost: Callable[[int], int], limit: int, avail: int) -> int:
    """Return the largest batch in ``[1, limit]`` whose cost fits, or 0.

    Binary search on the evaluated costs; the result is always a batch whose cost
    was checked. Batches of at least `GEMM_MULTIPLE` are rounded down to a multiple
    of it when that still fits.
    """
    if cost(1) > avail:
        return 0
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= avail:
            lo = mid
        else:
            hi = mid - 1
    rounded = lo - lo % GEMM_MULTIPLE
    if lo >= GEMM_MULTIPLE and rounded != lo and cost(rounded) <= avail:
        return rounded
    return lo


class _Planner:
    """Plans one sweep for a fixed prefetch depth."""

    def __init__(self, sites: Sequence[_Site], budgets: Budgets, prefetch: int) -> None:
        self.sites, self.budgets, self.prefetch = sites, budgets, prefetch
        self.n = len(sites)
        self.chi = sites[0].chi
        self.out_total = sum(self.chi * s.p * s.eta * s.e for s in sites)
        # Pinned ring of the site source, plus the host copy being read.
        self.site_ring = (prefetch + 2) * max(s.core_bytes for s in sites)
        # Two batches in and two out; full environments until batches are known.
        self.env_staging = 4 * max(s.env_bytes for s in sites[1:])

    def _costs(self, j: int, tiers: Sequence[Tier]) -> dict[str, Callable[[int], int]]:
        site, pf = self.sites[j], self.prefetch
        costs: dict[str, Callable[[int], int]] = {}
        if j < self.n - 1:
            staged_in = j > 0 and tiers[j] != "device"
            staged_out = tiers[j + 1] != "device"
            costs["env"] = lambda b: site.env(
                b, pf, staged_in=staged_in, staged_out=staged_out
            )
        if j > 0:
            staged = tiers[j] != "device"
            costs["sketch"] = lambda b: site.sketch(b, pf, staged=staged)
            costs["project"] = lambda b: site.project(b, pf)
        else:
            costs["first"] = lambda b: site.first(b, pf)
        return costs

    def _batches(self, tiers: Sequence[Tier], avail: int) -> list[dict[str, int]]:
        batches = []
        for j, site in enumerate(self.sites):
            if j > 0 and site.qr(self.prefetch) > avail:
                _fail(j, "QR", site.qr(self.prefetch), avail)
            chosen = {}
            for kernel, cost in self._costs(j, tiers).items():
                b = _largest_batch(cost, self.chi, avail)
                if b == 0:
                    _fail(j, kernel, cost(1), avail)
                chosen[kernel] = b
            batches.append(chosen)
        return batches

    def _peak(self, tiers: Sequence[Tier], batches: Sequence[dict[str, int]]) -> int:
        return max(
            cost(batches[j][kernel])
            for j in range(self.n)
            for kernel, cost in self._costs(j, tiers).items()
        )

    def _reserved(self, tiers: Sequence[Tier]) -> int:
        """Device bytes held outside the kernels.

        These are the device-tier environments and, on the CPU backend, where the
        device is the host, the output and the staging buffers as well.
        """
        resident = sum(
            s.env_bytes
            for j, s in enumerate(self.sites)
            if j > 0 and tiers[j] == "device"
        )
        if self.budgets.unified:
            spills = any(tier == "disk" for tier in tiers[1:])
            resident += self.out_total + self.site_ring + spills * self.env_staging
        return resident

    def _tiers(self, device_left: int, host_left: int) -> list[Tier]:
        """Assign the newest environments to the fastest tier that holds them."""
        tiers: list[Tier] = ["device"] * self.n
        level: Tier = "device"
        for j in range(self.n - 1, 0, -1):
            need = self.sites[j].env_bytes
            if level == "device" and device_left >= need:
                device_left -= need
                continue
            if level == "device":
                level = "disk" if self.budgets.unified else "host"
            if level == "host" and host_left >= need:
                tiers[j], host_left = "host", host_left - need
                continue
            tiers[j] = level = "disk"
        return tiers

    def plan(self) -> Plan:
        budgets, n = self.budgets, self.n
        staged: list[Tier] = ["device", *(["disk"] * (n - 1))]
        base = budgets.device - self._reserved(staged)
        batches = self._batches(staged, base)
        # Staged batches never grow in the final pass, so they bound the staging.
        self.env_staging = 4 * max(
            max(batches[j].values()) * s.a * s.e for j, s in enumerate(self.sites)
        )

        preferred = [
            {kernel: min(b, PREFERRED_BATCH) for kernel, b in chosen.items()}
            for chosen in batches
        ]
        host_left = budgets.host - self.out_total - self.site_ring - self.env_staging
        tiers = self._tiers(base - self._peak(staged, preferred), host_left)

        disk_bytes = sum(
            s.env_bytes for j, s in enumerate(self.sites) if tiers[j] == "disk"
        )
        if disk_bytes > budgets.disk:
            msg = (
                f"The environments need {disk_bytes} bytes on disk, but only "
                f"{budgets.disk} bytes are free in {budgets.scratch_dir}."
            )
            raise MemoryError(msg)

        reserved = self._reserved(tiers)
        batches = self._batches(tiers, budgets.device - reserved)
        host_env = sum(
            s.env_bytes for j, s in enumerate(self.sites) if tiers[j] == "host"
        )
        host_peak = self.out_total + self.site_ring
        if disk_bytes or host_env:
            host_peak += host_env + self.env_staging
        sites = tuple(
            SitePlan(
                env_batch=chosen.get("env", 0),
                sketch_batch=chosen.get("sketch", 0),
                project_batch=chosen.get("project", chosen.get("first", 0)),
                tier=tiers[j],
            )
            for j, chosen in enumerate(batches)
        )
        return Plan(
            sites,
            self.prefetch,
            device_peak=reserved + self._peak(tiers, batches),
            host_peak=host_peak,
            disk_bytes=disk_bytes,
        )


def _fail(j: int, kernel: str, need: int, avail: int) -> None:
    msg = (
        f"Site {j}: the {kernel} step needs {need} bytes with a batch of one, but "
        f"only {avail} bytes of the device budget are available. The site working "
        "set exceeds the budget."
    )
    raise MemoryError(msg)


def make_plan(
    site_shapes: Sequence[tuple[Shape, ...]],
    site_bytes: Sequence[int],
    chi_out: int,
    dtype: type | np.dtype,
    budgets: Budgets,
) -> Plan:
    """Plan the batches and the environment tiers of one sweep.

    Args:
        site_shapes: For every site, the padded ``(l, r, u, d)`` shape of each layer.
        site_bytes: For every site, the bytes of its cores.
        chi_out: The sketch size.
        dtype: The data type of the computation.
        budgets: The resolved budgets.

    Returns:
        The plan. Prefetching is dropped before giving up.

    Raises:
        MemoryError: If a site does not fit the device budget even with batches of
            one and no prefetching, or the environments do not fit on disk.
    """
    itemsize = np.dtype(dtype).itemsize
    n_sites = len(site_shapes)
    sites = [
        _Site(
            j,
            tuple(shapes),
            core_bytes,
            chi=chi_out,
            itemsize=itemsize,
            n_sites=n_sites,
        )
        for j, (shapes, core_bytes) in enumerate(zip(site_shapes, site_bytes))
    ]
    try:
        return _Planner(sites, budgets, prefetch=1).plan()
    except MemoryError:
        return _Planner(sites, budgets, prefetch=0).plan()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plan.py -o log_cli=false`
Expected: 25 passed.

- [ ] **Step 5: Check the reference plan**

Run:

```bash
uv run python - <<'EOF'
from pathlib import Path
import numpy as np
from src_method._plan import Budgets, make_plan
n, chi = 50, 2000
def site(j):
    def s(d):
        return (1 if j == 0 else d, 1 if j == n - 1 else d, 4, 4)
    return (s(4), s(4), s(4000), s(4))
shapes = [site(j) for j in range(n)]
sizes = [sum(int(np.prod(x)) * 16 for x in s) for s in shapes]
plan = make_plan(shapes, sizes, chi, np.complex128,
                 Budgets(36 * 10**9, 270 * 10**9, 450 * 10**9, Path("/tmp"), False))
print(plan.sites[25], plan.device_peak / 1e9, plan.disk_bytes / 1e9)
EOF
```

Expected (the spec's reference plan): `SitePlan(env_batch=128, sketch_batch=64, project_batch=16, tier='disk')`, a device peak of about 35 GB, and about 230 GB on disk.

- [ ] **Step 6: Lint**

```bash
uv run ruff format src/src_method/_plan.py tests/test_plan.py
uv run ruff check src/src_method/_plan.py tests/test_plan.py
```
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_plan.py, tests/test_plan.py
but commit -b feat/src-out-of-core -m $'feat(plan): ✨ plan batches and environment tiers to the budgets\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_plan.py`, `tests/test_plan.py`; never the untracked `.codegraph/`.

---

### Task 6: The environment store

**Files:**
- Create: `src/src_method/_store.py`
- Test: `tests/test_store.py` (create)

**Interfaces:**
- Consumes: `Plan`, `SitePlan` (Task 5); `current_stream`, `pinned_empty`, `to_device_async`, `to_host_async` (Task 1).
- Produces: `EnvironmentStore(plan, env_shapes, dtype, xp, scratch_dir, *, copy_stream)`, a context manager with
  - `put(j, lo, hi, x)`, `get(j, lo, hi) -> NDArray`, `prefetch(j, ranges)` where `ranges` is the ordered list of `(lo, hi)` that `get` will be called with, `drop(j)`, `close(*, wait=True)`, and `stall_seconds: float`;
  - `get` raises `RuntimeError("Batch {lo}:{hi} of environment {j} was read out of order.")`; a failed disk write re-raises as `OSError("Writing environment {j} to {path} failed ({strerror}); the plan spills ...")`;
  - module helpers `_write_all(fd, data, offset)` and `_read_all(fd, data, offset)`; `RING_SIZE = 2`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
"""Test the environment store on every tier, on the host backend."""

import errno

import numpy as np
import pytest

import src_method._store as store_module
from src_method._plan import Plan, SitePlan
from src_method._store import EnvironmentStore
from src_method.utils import NullStream

CHI = 10
SHAPES = [(CHI, 1), (CHI, 3, 2), (CHI, 4, 2), (CHI, 2, 1)]


def make_plan(tier, *, env=4, sketch=3):
    """Environments of sites 1-3 on ``tier``, written in batches of ``env``."""
    sites = [SitePlan(env, 0, 5, "device")]
    sites += [SitePlan(env, sketch, 5, tier) for _ in range(2)]
    sites += [SitePlan(0, sketch, 5, tier)]
    return Plan(tuple(sites), 1, 0, 0, 0)


def ranges(n, batch):
    return [(lo, min(lo + batch, n)) for lo in range(0, n, batch)]


def fill(store, rng, batch=4):
    """Put random environments for sites 1-3 and return them."""
    envs = {}
    for j in (1, 2, 3):
        envs[j] = rng.normal(size=SHAPES[j]) + 1j * rng.normal(size=SHAPES[j])
        for lo, hi in ranges(CHI, batch):
            store.put(j, lo, hi, envs[j][lo:hi])
    return envs


def open_store(tier, tmp_path, **batches):
    return EnvironmentStore(
        make_plan(tier, **batches),
        SHAPES,
        np.complex128,
        np,
        tmp_path,
        copy_stream=NullStream(),
    )


@pytest.mark.parametrize("tier", ["device", "host", "disk"])
def test_round_trip_in_other_batches(tier, tmp_path):
    rng = np.random.default_rng(0)
    with open_store(tier, tmp_path) as store:
        envs = fill(store, rng)
        for j in (3, 2, 1):
            batches = ranges(CHI, 3)  # read in batches of 3, written in batches of 4
            store.prefetch(j, batches)
            got = np.concatenate([store.get(j, lo, hi) for lo, hi in batches])
            np.testing.assert_array_equal(got, envs[j])
            store.drop(j)


def test_get_without_prefetch(tmp_path):
    rng = np.random.default_rng(1)
    with open_store("disk", tmp_path) as store:
        envs = fill(store, rng)

        np.testing.assert_array_equal(store.get(2, 7, 10), envs[2][7:10])


def test_get_out_of_order_raises(tmp_path):
    rng = np.random.default_rng(2)
    with open_store("host", tmp_path) as store:
        fill(store, rng)
        store.prefetch(1, [(0, 3), (3, 6)])

        with pytest.raises(RuntimeError, match="out of order"):
            store.get(1, 3, 6)


def test_disk_tier_files_are_removed(tmp_path):
    rng = np.random.default_rng(3)
    with open_store("disk", tmp_path) as store:
        fill(store, rng)
        (scratch,) = tmp_path.iterdir()
        store.prefetch(3, [(0, CHI)])
        store.get(3, 0, CHI)
        store.drop(3)
        assert sorted(p.name for p in scratch.iterdir()) == [
            "env-0001.bin",
            "env-0002.bin",
        ]
    assert list(tmp_path.iterdir()) == []


def fill_then_fail(tmp_path):
    with open_store("disk", tmp_path) as store:
        fill(store, np.random.default_rng(4))
        raise KeyError


def test_scratch_is_removed_after_an_error(tmp_path):
    with pytest.raises(KeyError):
        fill_then_fail(tmp_path)

    assert list(tmp_path.iterdir()) == []


def fill_then_read(tmp_path):
    with open_store("disk", tmp_path) as store:
        fill(store, np.random.default_rng(5))
        store.get(1, 0, 3)


def test_writer_errors_are_raised(monkeypatch, tmp_path):
    def full_disk(*_args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store_module, "_write_all", full_disk)

    with pytest.raises(OSError, match=r"env-0001\.bin failed \(No space left"):
        fill_then_read(tmp_path)

    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_store.py -o log_cli=false`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_method._store'`.

- [ ] **Step 3: Implement the store**

Create `src/src_method/_store.py`:

```python
"""Where the sweep keeps its environments: on the device, in host memory or on disk.

The right-to-left pass reads the environments in the reverse order of the
left-to-right pass that writes them, one batch of sketch columns at a time. Off the
device, every batch moves through a small ring of page-locked buffers: a writer
thread drains device-to-host copies into host arrays or files, and a reader thread
fills buffers ahead of use, so that transfers overlap the kernels.
"""

from __future__ import annotations

import os
import queue
import shutil
import uuid
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .utils import current_stream, pinned_empty, to_device_async, to_host_async

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path
    from types import ModuleType, TracebackType

    from numpy.typing import NDArray

    from ._plan import Plan

# Page-locked buffers per direction: one in use, one in flight.
RING_SIZE = 2
# How often a blocked reader checks whether the store is closing.
_POLL_SECONDS = 0.1


class _Closing(Exception):  # noqa: N818  (control flow, not an error)
    """Raised in the reader thread when the store closes under it."""


class EnvironmentStore:
    """Keep the environments ``C_j`` of a sweep, batch by batch.

    Use as a context manager: leaving it waits for pending writes, stops the
    threads and removes the scratch files, also after an exception.

    Args:
        plan: The plan of the sweep, which fixes the tier and batches of each site.
        env_shapes: The shape ``(chi, *left_bonds)`` of ``C_j`` for every site.
        dtype: The data type of the environments.
        xp: Array module (``numpy`` or ``cupy``).
        scratch_dir: Where the disk tier creates its per-process directory.
        copy_stream: The stream that performs the transfers.
    """

    def __init__(
        self,
        plan: Plan,
        env_shapes: Sequence[tuple[int, ...]],
        dtype: type | np.dtype,
        xp: ModuleType,
        scratch_dir: Path,
        *,
        copy_stream: Any,  # noqa: ANN401  (a cupy or null stream)
    ) -> None:
        self._plan = plan
        self._tiers = [site.tier for site in plan.sites]
        self._shapes = [tuple(shape) for shape in env_shapes]
        self._dtype = np.dtype(dtype)
        self._xp = xp
        self._copy = copy_stream
        self._device: dict[int, NDArray] = {}
        self._host: dict[int, np.ndarray] = {}
        self._fds: dict[int, int] = {}
        self._writes: dict[int, list[Future[None]]] = {}
        self._ahead: deque[tuple[int, int, int]] = deque()
        self._inflight: deque[tuple[int, int, int, Future[tuple[Any, np.ndarray]]]] = (
            deque()
        )
        self._closing = False
        self.stall_seconds = 0.0

        self._dir: Path | None = None
        if "disk" in self._tiers:
            self._dir = scratch_dir / f"src-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            self._dir.mkdir(parents=True)
            self._finalizer = weakref.finalize(
                self, shutil.rmtree, self._dir, ignore_errors=True
            )

        staged = [j for j, tier in enumerate(self._tiers) if j > 0 and tier != "device"]
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="src-env-writer")
        self._reader = ThreadPoolExecutor(1, thread_name_prefix="src-env-reader")
        self._out: queue.Queue[Any] = queue.Queue()
        self._in: queue.Queue[tuple[Any, Any]] = queue.Queue()
        if staged:
            n_bytes = max(self._batch_rows(j) * self._row_bytes(j) for j in staged)
            for _ in range(RING_SIZE):
                self._out.put(pinned_empty(n_bytes, xp))
                self._in.put((pinned_empty(n_bytes, xp), None))

    def _row_bytes(self, j: int) -> int:
        return prod(self._shapes[j][1:]) * self._dtype.itemsize

    def _batch_rows(self, j: int) -> int:
        sites = self._plan.sites
        return max(sites[j - 1].env_batch, sites[j].env_batch, sites[j].sketch_batch)

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store; see `close`."""
        self.close(wait=exc is None)

    def close(self, *, wait: bool = True) -> None:
        """Stop the threads and remove the scratch files.

        Args:
            wait: Wait for the pending writes and re-raise their first error.
        """
        try:
            if wait:
                for futures in self._writes.values():
                    for future in futures:
                        future.result()
        finally:
            self._closing = True
            self._writer.shutdown(wait=True, cancel_futures=True)
            self._reader.shutdown(wait=True, cancel_futures=True)
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._device.clear()
            self._host.clear()
            if self._dir is not None:
                self._finalizer()

    # --- writing ---

    def put(self, j: int, lo: int, hi: int, x: NDArray) -> None:
        """Store columns ``lo:hi`` of ``C_j``.

        Args:
            j: The site.
            lo: The first column.
            hi: One past the last column.
            x: The columns, ``(hi - lo, *left_bonds)``, on the device.
        """
        if self._tiers[j] == "device":
            if j not in self._device:
                self._device[j] = self._xp.empty(self._shapes[j], dtype=self._dtype)
            self._device[j][lo:hi] = x
            return
        self._raise_failed()
        self._open(j)
        start = perf_counter()
        buffer = self._out.get()
        self.stall_seconds += perf_counter() - start
        host = buffer[: x.nbytes].view(self._dtype).reshape(x.shape)
        # The copy starts once the kernel that wrote x is done.
        self._copy.wait_event(current_stream(self._xp).record())
        to_host_async(x, host, self._copy)
        done = self._copy.record()
        # x stays referenced by the task until the copy is complete.
        future = self._writer.submit(
            self._write, j, lo, x, host=host, done=done, buffer=buffer
        )
        self._writes.setdefault(j, []).append(future)

    def _open(self, j: int) -> None:
        if self._tiers[j] == "host" and j not in self._host:
            self._host[j] = np.empty(self._shapes[j], dtype=self._dtype)
        elif self._tiers[j] == "disk" and j not in self._fds:
            fd = os.open(self._path(j), os.O_RDWR | os.O_CREAT, 0o600)
            self._fds[j] = fd
            os.ftruncate(fd, prod(self._shapes[j]) * self._dtype.itemsize)

    def _path(self, j: int) -> Path:
        assert self._dir is not None  # noqa: S101  (the disk tier creates it)
        return self._dir / f"env-{j:04d}.bin"

    def _write(
        self,
        j: int,
        lo: int,
        x: NDArray,
        *,
        host: np.ndarray,
        done: Any,  # noqa: ANN401  (a cupy or null event)
        buffer: Any,  # noqa: ANN401  (a page-locked buffer)
    ) -> None:
        try:
            done.synchronize()
            del x
            if self._tiers[j] == "host":
                np.copyto(self._host[j][lo : lo + host.shape[0]], host)
                return
            offset = lo * self._row_bytes(j)
            try:
                _write_all(self._fds[j], memoryview(buffer[: host.nbytes]), offset)
            except OSError as err:
                msg = (
                    f"Writing environment {j} to {self._path(j)} failed "
                    f"({err.strerror}); the plan spills {self._plan.disk_bytes} bytes "
                    f"to {self._dir}."
                )
                raise OSError(err.errno, msg) from err
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    self._fds[j], offset, host.nbytes, os.POSIX_FADV_DONTNEED
                )
        finally:
            self._out.put(buffer)

    def _raise_failed(self) -> None:
        """Re-raise the first error of a finished write, dropping finished writes."""
        for j, futures in self._writes.items():
            pending = []
            for future in futures:
                if not future.done():
                    pending.append(future)
                elif (error := future.exception()) is not None:
                    raise error
            self._writes[j] = pending

    # --- reading ---

    def prefetch(self, j: int, ranges: Iterable[tuple[int, int]]) -> None:
        """Announce the batches of ``C_j`` that `get` will be asked for, in order.

        Off the device, the reader starts filling buffers with the first ones.

        Args:
            j: The site.
            ranges: The ``(lo, hi)`` column ranges, in the order they will be read.
        """
        if self._tiers[j] == "device":
            return
        self._ahead.extend((j, lo, hi) for lo, hi in ranges)
        self._schedule()

    def _schedule(self) -> None:
        while self._ahead and len(self._inflight) < RING_SIZE:
            j, lo, hi = self._ahead.popleft()
            writes = list(self._writes.get(j, []))
            future = self._reader.submit(self._read, j, lo, hi, writes)
            self._inflight.append((j, lo, hi, future))

    def _read(
        self, j: int, lo: int, hi: int, writes: Sequence[Future[None]]
    ) -> tuple[Any, np.ndarray]:
        for write in writes:
            write.result()
        while True:
            try:
                buffer, event = self._in.get(timeout=_POLL_SECONDS)
                break
            except queue.Empty:
                if self._closing:
                    raise _Closing from None
        if event is not None:
            event.synchronize()
        shape = (hi - lo, *self._shapes[j][1:])
        n_bytes = prod(shape) * self._dtype.itemsize
        host = buffer[:n_bytes].view(self._dtype).reshape(shape)
        if self._tiers[j] == "host":
            np.copyto(host, self._host[j][lo:hi])
        else:
            _read_all(
                self._fds[j], memoryview(buffer[:n_bytes]), lo * self._row_bytes(j)
            )
        return buffer, host

    def get(self, j: int, lo: int, hi: int) -> NDArray:
        """Return columns ``lo:hi`` of ``C_j`` on the device.

        Args:
            j: The site.
            lo: The first column.
            hi: One past the last column.

        Returns:
            The columns, ``(hi - lo, *left_bonds)``. Kernels on the current stream
            see them once their transfer is done.

        Raises:
            RuntimeError: If the batch is not the next one announced by `prefetch`.
        """
        if self._tiers[j] == "device":
            return self._device[j][lo:hi]
        self._raise_failed()
        if not self._inflight and not self._ahead:
            self.prefetch(j, [(lo, hi)])
        if not self._inflight or self._inflight[0][:3] != (j, lo, hi):
            msg = f"Batch {lo}:{hi} of environment {j} was read out of order."
            raise RuntimeError(msg)
        future = self._inflight.popleft()[3]
        start = perf_counter()
        buffer, host = future.result()
        self.stall_seconds += perf_counter() - start
        # On the current stream, so that the copy is ordered with the kernels that
        # use it and with the ones that freed the memory it reuses.
        stream = current_stream(self._xp)
        device = to_device_async(host, self._xp, stream)
        self._in.put((buffer, stream.record()))
        self._schedule()
        return device

    def drop(self, j: int) -> None:
        """Release ``C_j``: its memory, or its file."""
        for future in self._writes.pop(j, []):
            future.result()
        self._device.pop(j, None)
        self._host.pop(j, None)
        fd = self._fds.pop(j, None)
        if fd is not None:
            os.close(fd)
            self._path(j).unlink()


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    """Write all of ``data`` at ``offset``; `os.pwrite` may write less."""
    while data:
        written = os.pwrite(fd, data, offset)
        data, offset = data[written:], offset + written


def _read_all(fd: int, data: memoryview, offset: int) -> None:
    """Fill ``data`` from ``offset``; `os.preadv` may read less."""
    while data:
        read = os.preadv(fd, [data], offset)
        if read == 0:
            msg = f"Unexpected end of file at offset {offset}."
            raise OSError(msg)
        data, offset = data[read:], offset + read
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -o log_cli=false`
Expected: 8 passed, with no `ResourceWarning` or thread warnings.

- [ ] **Step 5: Lint**

```bash
uv run ruff format src/src_method/_store.py tests/test_store.py
uv run ruff check src/src_method/_store.py tests/test_store.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_store.py, tests/test_store.py
but commit -b feat/src-out-of-core -m $'feat(store): ✨ keep environments on the device, in host memory or on disk\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_store.py`, `tests/test_store.py`; never the untracked `.codegraph/`.

---

### Task 7: The site source

**Files:**
- Create: `src/src_method/_sites.py`
- Test: `tests/test_sites.py` (create; Task 8 appends the end-to-end tests)

**Interfaces:**
- Consumes: `infer_kind`, `pad_site`, `padded_shape` (Task 2); `NullEvent`, `current_stream`, `is_host`, `pinned_empty`, `to_device_async` (Task 1).
- Produces:
  - `padded_shapes(layers) -> list[tuple[tuple[int, ...], ...]]` and `site_bytes(layers) -> list[int]`, from `.shape` and `.dtype` only.
  - `SiteSource(layers, xp, *, depth)`, a context manager with `len()`, `source[j] -> tuple[NDArray, ...]` (padded cores of site `j` on the device), `prefetch(j)` (no-op out of range or with `depth == 0`), `stall_seconds: float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sites.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_sites.py -o log_cli=false`
Expected: FAIL with `ModuleNotFoundError: No module named 'src_method._sites'`.

- [ ] **Step 3: Implement the site source**

Create `src/src_method/_sites.py`:

```python
"""Lazy, per-site access to the cores of a stack.

A core is read from its source (a NumPy array, an `np.memmap`, a zarr or HDF5
dataset) only when the sweep reaches its site, and a background thread can read the
next site while the current one is computed. Shapes come from ``.shape`` alone, so
planning never reads any data.
"""

from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ._tensor_train import infer_kind, pad_site, padded_shape
from .utils import NullEvent, current_stream, is_host, pinned_empty, to_device_async

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType, TracebackType

    from numpy.typing import NDArray

# Byte alignment of each core inside a page-locked staging buffer.
_ALIGN = 256


def padded_shapes(layers: Sequence[Sequence[Any]]) -> list[tuple[tuple[int, ...], ...]]:
    """Return, for every site, the padded ``(l, r, u, d)`` shape of each layer.

    Args:
        layers: A stack in ket form.

    Returns:
        One tuple of shapes per site, without reading any data.
    """
    kinds = [infer_kind(layer) for layer in layers]
    last = len(layers[0]) - 1
    return [
        tuple(
            padded_shape(tuple(layer[j].shape), kind, j, last)
            for layer, kind in zip(layers, kinds)
        )
        for j in range(last + 1)
    ]


def site_bytes(layers: Sequence[Sequence[Any]]) -> list[int]:
    """Return, for every site, the bytes of its cores over all layers."""
    return [
        sum(
            prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize for layer in layers
        )
        for j in range(len(layers[0]))
    ]


def _aligned(n_bytes: int) -> int:
    return -(-n_bytes // _ALIGN) * _ALIGN


class SiteSource:
    """Padded device cores of a stack, one site at a time.

    Reading from the source and staging into page-locked memory happen on a
    background thread; the host-to-device copy is issued on the current stream when
    the site is requested, so it is ordered with the kernels that use it.

    Use as a context manager, so that the loader thread stops.

    Args:
        layers: A stack in ket form, each layer a sequence of array-likes with
            ``shape``, ``dtype`` and ``np.asarray`` support.
        xp: Array module (``numpy`` or ``cupy``).
        depth: How many sites `prefetch` may load ahead (0 disables the thread).
    """

    def __init__(
        self, layers: Sequence[Sequence[Any]], xp: ModuleType, *, depth: int
    ) -> None:
        self._layers = layers
        self._kinds = [infer_kind(layer) for layer in layers]
        self._last = len(layers[0]) - 1
        self._xp = xp
        self._pending: dict[int, Future[tuple[list[Any], Any]]] = {}
        self._executor = (
            ThreadPoolExecutor(1, thread_name_prefix="src-site-loader")
            if depth > 0
            else None
        )
        self._buffers: queue.Queue[tuple[Any, Any]] = queue.Queue()
        self.stall_seconds = 0.0
        if not is_host(xp):
            n_bytes = max(
                sum(
                    _aligned(prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize)
                    for layer in layers
                )
                for j in range(self._last + 1)
            )
            for _ in range(depth + 1):
                self._buffers.put((pinned_empty(n_bytes, xp), NullEvent()))

    def __len__(self) -> int:
        """Return the number of sites."""
        return self._last + 1

    def __enter__(self) -> Self:
        """Return the source."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the loader thread."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def prefetch(self, j: int) -> None:
        """Start reading site ``j`` in the background, if there is such a site."""
        if self._executor is None or not 0 <= j <= self._last or j in self._pending:
            return
        self._pending[j] = self._executor.submit(self._stage, j)

    def __getitem__(self, j: int) -> tuple[NDArray, ...]:
        """Return the padded cores of site ``j`` on the device."""
        future = self._pending.pop(j, None)
        if future is None:
            staged, buffer = self._stage(j)
        else:
            start = perf_counter()
            staged, buffer = future.result()
            self.stall_seconds += perf_counter() - start
        if is_host(self._xp):
            return tuple(staged)
        stream = current_stream(self._xp)
        cores = [
            core
            if isinstance(core, self._xp.ndarray)
            else to_device_async(core, self._xp, stream)
            for core in staged
        ]
        # The buffer is refilled only once these copies are done.
        self._buffers.put((buffer, stream.record()))
        return tuple(
            pad_site(core, kind, j, self._last)
            for core, kind in zip(cores, self._kinds)
        )

    def _stage(self, j: int) -> tuple[list[Any], Any]:
        """Read site ``j`` into memory.

        Returns padded arrays on the host backend, and on the GPU unpadded
        page-locked copies (or the device arrays given as input) with their buffer.
        """
        if is_host(self._xp):
            padded = [
                pad_site(_read(layer[j]), kind, j, self._last)
                for layer, kind in zip(self._layers, self._kinds)
            ]
            return padded, None
        buffer, previous = self._buffers.get()
        previous.synchronize()
        staged, offset = [], 0
        for layer in self._layers:
            core = layer[j]
            if not isinstance(core, self._xp.ndarray):
                host = np.asarray(core)
                view = buffer[offset : offset + host.nbytes]
                view = view.view(host.dtype).reshape(host.shape)
                np.copyto(view, host)
                offset += _aligned(host.nbytes)
                core = view
            staged.append(core)
        return staged, buffer


def _read(core: Any) -> NDArray:  # noqa: ANN401  (any array-like)
    """Bring a core into memory; memmaps are read now, not on first touch."""
    if isinstance(core, np.memmap) or not isinstance(core, np.ndarray):
        return np.array(core)
    return core
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sites.py -o log_cli=false`
Expected: 3 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff format src/src_method/_sites.py tests/test_sites.py
uv run ruff check src/src_method/_sites.py tests/test_sites.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_sites.py, tests/test_sites.py
but commit -b feat/src-out-of-core -m $'feat(sites): ✨ read the cores of a stack one site at a time\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_sites.py`, `tests/test_sites.py`; never the untracked `.codegraph/`.

---

### Task 8: The out-of-core driver and the `resources` keyword

**Files:**
- Modify: `src/src_method/_sweep.py` (replace with the driver)
- Modify: `src/src_method/stack.py`, `src/src_method/apply.py`, `src/src_method/compress.py`
- Test: `tests/test_stack.py` (integration tests), `tests/test_sites.py` (append end-to-end tests)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `sweep(layers, kind, chi_out, prng, xp, *, cutoff=0.0, dtype=np.float64, resources=None) -> list[NDArray]` (host arrays); logs `SRC plan` (fields `prefetch`, `device_peak_bytes`, `host_peak_bytes`, `disk_bytes`, `scratch_dir`, `tiers`, `batches`) and `SRC stalls` (`site_seconds`, `environment_seconds`) at `info`, the per-pass times and `Device pool` (the pool high-water mark) at `debug`.
  - `src(..., resources: Resources | None = None)`, and the same keyword on `apply` and `compress`, passed through.

- [ ] **Step 1: Write the failing tests**

Save as `/tmp/t8_test_stack.diff` and apply from the repository root with `git apply /tmp/t8_test_stack.diff` (it only touches the working tree):

```diff
--- a/tests/test_stack.py
+++ b/tests/test_stack.py
@@ -5,7 +5,9 @@
 import numpy as np
 import pytest

-from src_method import apply, compress, src
+import src_method._sweep as sweep_module
+from src_method import Resources, apply, compress, src
+from src_method._plan import make_plan
 from src_method._sweep import sweep

 # -------------
@@ -346,3 +348,58 @@
     np.testing.assert_allclose(
         ref.distance(qtn.MatrixProductState(out)), 0.0, atol=1e-6
     )
+
+
+# ----------------------------
+# --- Budgets and spilling ---
+# ----------------------------
+
+
+@pytest.fixture
+def plans(monkeypatch):
+    """Record the plan of every sweep."""
+    recorded = []
+
+    def spy(*args, **kwargs):
+        recorded.append(make_plan(*args, **kwargs))
+        return recorded[-1]
+
+    monkeypatch.setattr(sweep_module, "make_plan", spy)
+    return recorded
+
+
+def test_tiny_budget_batches_and_spills(rng, tmp_path, plans):
+    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]
+    tight = Resources(host_memory="1MB", scratch_dir=tmp_path)
+
+    default = src(*stack, chi_out=16, seed=3, dtype=np.complex128)
+    spilled = src(*stack, chi_out=16, seed=3, dtype=np.complex128, resources=tight)
+
+    assert all(site.tier == "device" for site in plans[0].sites)
+    assert "disk" in {site.tier for site in plans[1].sites}
+    assert min(site.sketch_batch for site in plans[1].sites[1:]) < 16
+    # Rounding differs, and the output cores of an ill-conditioned sketch with it,
+    # but not the operator they represent.
+    assert rel_error(spilled, dense(default)) < 1e-10
+    assert list(tmp_path.iterdir()) == []
+
+
+def test_budget_too_small_raises(rng):
+    stack = make_stack("A B psi", rng)
+
+    with pytest.raises(MemoryError, match="working set exceeds the budget"):
+        src(*stack, chi_out=8, resources=Resources(host_memory="1kB"))
+
+
+def test_apply_passes_resources(rng):
+    A, psi = make_stack("A psi", rng)
+
+    with pytest.raises(MemoryError, match="working set exceeds the budget"):
+        apply(A, psi, chi_out=8, resources=Resources(host_memory="1kB"))
+
+
+def test_compress_passes_resources(rng):
+    (A,) = make_stack("A", rng)
+
+    with pytest.raises(MemoryError, match="working set exceeds the budget"):
+        compress(A, chi_out=8, resources=Resources(host_memory="1kB"))
```

Append to `tests/test_sites.py` (and add `from src_method import src` to its imports):

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stack.py tests/test_sites.py -o log_cli=false -m "not perf"`
Expected: 4 failures and 1 error: `TypeError: src() got an unexpected keyword argument 'resources'` (and the same for `apply` and `compress`), an `AttributeError` for `make_plan` on `src_method._sweep` in the `plans` fixture, and `test_src_reads_each_site_once_per_pass` failing because today's sweep reads every site up front.

- [ ] **Step 3: Replace the sweep with the driver**

Replace `src/src_method/_sweep.py` with:

```python
"""The generic SRC sweep over a stack of tensor trains.

One kernel covers every stack in ket form: ``k`` MPOs, optionally followed by an
MPS. Sites are padded to bulk ``(l, r, u, d)`` views (see
`src_method._tensor_train.pad`), so a single set of einsum equations, generated
from the depth ``k``, serves the boundaries and the bulk alike.

The left-to-right sweep sketches the open physical legs with Gaussian ``omega``
tensors and accumulates the environments ``C``; the sketch index is shared by
every site (a Khatri-Rao sketch). The right-to-left sweep builds the output
through `truncated_qr` while carrying the projected environment ``S``.

Every contraction runs in batches sized by `src_method._plan.make_plan` to the
memory budgets. The cores are read one site at a time (`SiteSource`) and the
environments live on the device, in host memory or on disk (`EnvironmentStore`).
"""

from __future__ import annotations

from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ._kernels import SiteKernels
from ._plan import make_plan, resolve_budgets
from ._sites import SiteSource, padded_shapes, site_bytes
from ._store import EnvironmentStore
from ._tensor_train import unpad
from .utils import (
    device_pool_bytes,
    device_pool_limit,
    new_stream,
    to_numpy,
    truncated_qr,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

    from ._plan import Plan, Resources
    from ._tensor_train import TrainKind

logger = structlog.get_logger(__name__)


def _ranges(n: int, batch: int) -> Iterator[tuple[int, int]]:
    """Split ``range(n)`` into consecutive ``(lo, hi)`` batches."""
    for lo in range(0, n, batch):
        yield lo, min(lo + batch, n)


def sweep(
    layers: Sequence[Sequence[NDArray]],
    kind: TrainKind,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    *,
    cutoff: float = 0.0,
    dtype: type = np.float64,
    resources: Resources | None = None,
) -> list[NDArray]:
    """Contract and compress a stack in ket form with one SRC sweep.

    Args:
        layers: MPOs, optionally followed by one MPS, all with the same number
            (at least three) of sites and matching physical legs. Sites may be any
            array-likes with ``shape``, ``dtype`` and ``np.asarray`` support; each is
            read only when the sweep reaches it.
        kind: The kind of the contracted train.
        chi_out: The sketch size, which is the maximum output bond dimension.
        prng: The generator for the Gaussian sketches, always host-side so that a
            seed gives the same draws on every device.
        xp: Array module (``numpy`` or ``cupy``).
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        dtype: The data type of the sketches.
        resources: Memory budgets and scratch space; detected when ``None``.

    Returns:
        The site arrays of the compressed train in right-canonical form, as numpy
        arrays.
    """
    shapes = padded_shapes(layers)
    # The dtype of the environments and the output: the sketches promoted by the
    # cores, as the contractions would.
    work = np.result_type(dtype, *(layer[0].dtype for layer in layers))
    budgets = resolve_budgets(resources, xp)
    plan = make_plan(shapes, site_bytes(layers), chi_out, work, budgets)
    logger.info(
        "SRC plan",
        prefetch=plan.prefetch,
        device_peak_bytes=plan.device_peak,
        host_peak_bytes=plan.host_peak,
        disk_bytes=plan.disk_bytes,
        scratch_dir=str(budgets.scratch_dir) if plan.disk_bytes else None,
        tiers=[site.tier for site in plan.sites],
        batches=[
            (site.env_batch, site.sketch_batch, site.project_batch)
            for site in plan.sites
        ],
    )
    env_shapes = [(chi_out, *(s[0] for s in site)) for site in shapes]
    kernels = SiteKernels(len(layers))
    with (
        device_pool_limit(xp, budgets.device),
        SiteSource(layers, xp, depth=plan.prefetch) as source,
        EnvironmentStore(
            plan, env_shapes, work, xp, budgets.scratch_dir, copy_stream=new_stream(xp)
        ) as store,
    ):
        tms = perf_counter_ns()
        _left_to_right(
            kernels, source, store, plan, chi_out=chi_out, prng=prng, xp=xp, dtype=dtype
        )
        logger.debug("Left-to-right sweep", seconds=(perf_counter_ns() - tms) * 1e-9)
        tms = perf_counter_ns()
        eta = _right_to_left(
            kernels,
            source,
            store,
            plan,
            chi_out=chi_out,
            cutoff=cutoff,
            xp=xp,
            dtype=work,
        )
        logger.debug("Right-to-left sweep", seconds=(perf_counter_ns() - tms) * 1e-9)
        logger.info(
            "SRC stalls",
            site_seconds=source.stall_seconds,
            environment_seconds=store.stall_seconds,
        )
        logger.debug("Device pool", bytes=device_pool_bytes(xp))
    return [to_numpy(site) for site in unpad(eta, kind)]


def _left_to_right(
    kernels: SiteKernels,
    source: SiteSource,
    store: EnvironmentStore,
    plan: Plan,
    *,
    chi_out: int,
    prng: np.random.Generator,
    xp: ModuleType,
    dtype: type,
) -> None:
    """Build the environments ``C_1 .. C_{n-1}`` into the store."""
    depth = len(kernels.eqs.ltr.split(",")) - 2
    first_env = xp.ones((chi_out,) + (1,) * depth, dtype=dtype)
    n_sites = len(source)
    for j in range(n_sites - 1):
        cores = source[j]
        if plan.prefetch:
            source.prefetch(j + 1)
        up, down = cores[0].shape[2], cores[-1].shape[3]
        omega = xp.asarray(prng.normal(size=(chi_out, up, down))).astype(dtype)
        batches = list(_ranges(chi_out, plan.sites[j].env_batch))
        if j > 0:
            store.prefetch(j, batches)
        for lo, hi in batches:
            env = first_env[lo:hi] if j == 0 else store.get(j, lo, hi)
            store.put(j + 1, lo, hi, kernels.env(env, omega[lo:hi], cores))


def _right_to_left(
    kernels: SiteKernels,
    source: SiteSource,
    store: EnvironmentStore,
    plan: Plan,
    *,
    chi_out: int,
    cutoff: float,
    xp: ModuleType,
    dtype: np.dtype,
) -> list[NDArray]:
    """Build the output cores, host-side, from the last site to the first.

    ``dtype`` is the working dtype of the sweep, that of the environments.
    """
    depth = len(kernels.eqs.ltr.split(",")) - 2
    n_sites = len(source)
    eta_reversed: list[NDArray] = []
    proj = xp.ones((1,) * (depth + 1), dtype=dtype)
    for j in range(n_sites - 1, 0, -1):
        cores = source[j]
        if plan.prefetch:
            source.prefetch(j - 1)
        site = plan.sites[j]
        batches = list(_ranges(chi_out, site.sketch_batch))
        store.prefetch(j, batches)
        up, down = cores[0].shape[2], cores[-1].shape[3]
        sketch = xp.empty((proj.shape[0], up, down, chi_out), dtype=dtype)
        for lo, hi in batches:
            sketch[..., lo:hi] = kernels.sketch(store.get(j, lo, hi), cores, proj)
        store.drop(j)
        rows = sketch.shape[0] * up * down
        Q = truncated_qr(sketch.reshape(rows, chi_out), cutoff, xp)
        del sketch
        eta_j = Q.reshape(proj.shape[0], up, down, Q.shape[1]).transpose(3, 0, 1, 2)
        new_proj = xp.empty((Q.shape[1], *(c.shape[0] for c in cores)), dtype=dtype)
        for lo, hi in _ranges(Q.shape[1], site.project_batch):
            new_proj[lo:hi] = kernels.project(eta_j[lo:hi].conj(), cores, proj)
        proj = new_proj
        eta_reversed.append(to_numpy(eta_j))
    cores = source[0]
    up, down = cores[0].shape[2], cores[-1].shape[3]
    first = xp.empty((proj.shape[0], up, down), dtype=dtype)
    for lo, hi in _ranges(proj.shape[0], plan.sites[0].project_batch):
        first[lo:hi] = kernels.first(cores, proj[lo:hi]).reshape(hi - lo, up, down)
    return [to_numpy(first)[None], *reversed(eta_reversed)]
```

- [ ] **Step 4: Pass `resources` through the public functions**

Save as `/tmp/t8_stack.diff` and apply from the repository root with `git apply /tmp/t8_stack.diff` (it only touches the working tree):

```diff
--- a/src/src_method/stack.py
+++ b/src/src_method/stack.py
@@ -26,6 +26,8 @@

     from numpy.typing import NDArray

+    from ._plan import Resources
+
 setup_logging()
 logger = structlog.get_logger(__name__)

@@ -42,6 +44,7 @@
     dtype: type = np.float64,
     seed: int | None = None,
     device: str = "cpu",
+    resources: Resources | None = None,
 ) -> list[NDArray]:
     """Contract a stack of tensor trains and compress the result with SRC.

@@ -79,6 +82,9 @@
         seed: An optional seed for the random number generator.
         device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy). Requires
             the optional ``cupy`` dependency for GPU execution.
+        resources: Memory budgets and scratch space for the sweep (see
+            `Resources`); every budget left unset is detected. Ignored for
+            two-site stacks.

     Returns:
         The site arrays of the compressed train (MPS or MPO), in right-canonical
@@ -109,6 +115,15 @@
         output=kind,
         device=xp.__name__,
     )
-    result = sweep(layers, kind, chi_out, prng, xp, cutoff=cutoff, dtype=dtype)
+    result = sweep(
+        layers,
+        kind,
+        chi_out,
+        prng,
+        xp,
+        cutoff=cutoff,
+        dtype=dtype,
+        resources=resources,
+    )
     logger.info("SRC complete.")
     return result
```

Save as `/tmp/t8_apply.diff` and apply from the repository root with `git apply /tmp/t8_apply.diff` (it only touches the working tree):

```diff
--- a/src/src_method/apply.py
+++ b/src/src_method/apply.py
@@ -20,6 +20,8 @@

     from numpy.typing import NDArray

+    from ._plan import Resources
+

 def apply(
     left_tensor: Sequence[NDArray],
@@ -30,6 +32,7 @@
     dtype: type = np.float64,
     seed: int | None = None,
     device: str = "cpu",
+    resources: Resources | None = None,
 ) -> list[NDArray]:
     """Applies the Successive Randomized Compression (SRC) algorithm.

@@ -55,6 +58,7 @@
         seed: An optional seed for the random number generator.
         device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy).  Requires
             the optional ``cupy`` dependency for GPU execution.
+        resources: Memory budgets and scratch space; see `src_method.stack.src`.

     Returns:
         The site arrays of the compressed tensor network (MPS or MPO).
@@ -83,4 +87,5 @@
         dtype=dtype,
         seed=seed,
         device=device,
+        resources=resources,
     )
```

Save as `/tmp/t8_compress.diff` and apply from the repository root with `git apply /tmp/t8_compress.diff` (it only touches the working tree):

```diff
--- a/src/src_method/compress.py
+++ b/src/src_method/compress.py
@@ -19,6 +19,8 @@

     from numpy.typing import NDArray

+    from ._plan import Resources
+

 def compress(
     tensor: Sequence[NDArray],
@@ -28,6 +30,7 @@
     dtype: type = np.float64,
     seed: int | None = None,
     device: str = "cpu",
+    resources: Resources | None = None,
 ) -> list[NDArray]:
     """Applies the Successive Randomized Compression (SRC) algorithm.

@@ -51,6 +54,7 @@
         seed: An optional seed for the random number generator.
         device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy).  Requires
             the optional ``cupy`` dependency for GPU execution.
+        resources: Memory budgets and scratch space; see `src_method.stack.src`.

     Returns:
         The site arrays of the compressed tensor network (MPS or MPO).
@@ -62,5 +66,11 @@
         ImportError: If ``device="gpu"`` but cupy is not installed.
     """
     return src(
-        tensor, chi_out=chi_out, cutoff=cutoff, dtype=dtype, seed=seed, device=device
+        tensor,
+        chi_out=chi_out,
+        cutoff=cutoff,
+        dtype=dtype,
+        seed=seed,
+        device=device,
+        resources=resources,
     )
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -o log_cli=false`
Expected: all pass (the GPU module skips without CuPy); the two `perf` benchmarks within noise of their times before this task.

- [ ] **Step 6: Lint**

```bash
uv run ruff format src tests
uv run ruff check src tests
```
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
but diff
# Pick the IDs of exactly these files: src/src_method/_sweep.py, src/src_method/stack.py, src/src_method/apply.py, src/src_method/compress.py, tests/test_stack.py, tests/test_sites.py
but commit -b feat/src-out-of-core -m $'feat(sweep): ✨ out-of-core SRC sweep planned to memory budgets\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `src/src_method/_sweep.py`, `src/src_method/stack.py`, `src/src_method/apply.py`, `src/src_method/compress.py`, `tests/test_stack.py`, `tests/test_sites.py`; never the untracked `.codegraph/`.

---

### Task 9: GPU tests

**Files:**
- Modify: `tests/test_gpu_backend.py` (imports; append helpers and two tests)

**Interfaces:**
- Consumes: `Resources`, `src`, `SiteKernels`, `Budgets`, `Plan`, `make_plan`, `padded_shapes`, `site_bytes`.
- Produces: nothing for later tasks.

The GPU budgets are derived from a plan made with `make_plan` (a pure function), so that the run reaches every tier: half the roomy device peak and the roomy host peak plus five environments give `device`, `host` and `disk` tiers and batches of 32 for this stack.

- [ ] **Step 1: Write the tests**

Save as `/tmp/t9_test_gpu.diff` and apply from the repository root with `git apply /tmp/t9_test_gpu.diff` (it only touches the working tree):

```diff
--- a/tests/test_gpu_backend.py
+++ b/tests/test_gpu_backend.py
@@ -9,7 +9,11 @@
 import pytest
 import quimb.tensor as qtn

-from src_method import apply, compress, src
+import src_method._sweep as sweep_module
+from src_method import Resources, apply, compress, src
+from src_method._kernels import SiteKernels
+from src_method._plan import Budgets, Plan, make_plan
+from src_method._sites import padded_shapes, site_bytes

 cupy = pytest.importorskip("cupy")

@@ -181,3 +185,97 @@
     ref = H1.apply(H2.apply(psi, compress=False), compress=False)

     np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)
+
+
+# ---------------------------------------
+# --- Budgets, batching and spilling ---
+# ---------------------------------------
+
+
+def random_mpo_arrays(bonds: list[int], rng: np.random.Generator) -> list[np.ndarray]:
+    """Complex Gaussian MPO with the given bonds and physical legs of 2."""
+
+    def site(*shape: int) -> np.ndarray:
+        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
+
+    lefts, rights = [None, *bonds], [*bonds, None]
+    return [
+        site(*(b for b in (lb, rb) if b is not None), 2, 2)
+        for lb, rb in zip(lefts, rights)
+    ]
+
+
+def dense_mpo(train: list[np.ndarray]) -> np.ndarray:
+    """Contract an MPO into a ``(U, D)`` matrix."""
+    T = train[0]
+    for W in train[1:-1]:
+        T = np.einsum("aUD,abud->bUuDd", T, W)
+        T = T.reshape(W.shape[1], T.shape[1] * T.shape[2], T.shape[3] * T.shape[4])
+    T = np.einsum("aUD,aud->UuDd", T, train[-1])
+    return T.reshape(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3])
+
+
+@pytest.fixture
+def plans(monkeypatch: pytest.MonkeyPatch) -> list[Plan]:
+    """Record the plan of every sweep."""
+    recorded: list[Plan] = []
+
+    def spy(*args: object, **kwargs: object) -> Plan:
+        recorded.append(make_plan(*args, **kwargs))
+        return recorded[-1]
+
+    monkeypatch.setattr(sweep_module, "make_plan", spy)
+    return recorded
+
+
+def test_gpu_tiers_and_batches_match_cpu(tmp_path, plans: list[Plan]) -> None:
+    """Every tier and small batches on the GPU give the operator of the CPU run."""
+    rng = np.random.default_rng(7)
+    stack = [random_mpo_arrays([4, 8, 8, 8, 4], rng) for _ in range(4)]
+    chi = 64
+    shapes, sizes = padded_shapes(stack), site_bytes(stack)
+    roomy = make_plan(
+        shapes,
+        sizes,
+        chi,
+        np.complex128,
+        Budgets(10**10, 10**10, 10**12, tmp_path, unified=False),
+    )
+    env = chi * 8**4 * 16  # one bulk environment
+    tight = Resources(
+        gpu_memory=roomy.device_peak // 2,
+        host_memory=roomy.host_peak + 5 * env,
+        scratch_dir=tmp_path,
+    )
+
+    cpu = src(*stack, chi_out=chi, seed=3, dtype=np.complex128)
+    gpu = src(
+        *stack, chi_out=chi, seed=3, dtype=np.complex128, device="gpu", resources=tight
+    )
+
+    assert {site.tier for site in plans[-1].sites} == {"device", "host", "disk"}
+    assert min(site.sketch_batch for site in plans[-1].sites[1:]) < chi
+    reference = dense_mpo(cpu)
+    error = np.linalg.norm(dense_mpo(gpu) - reference) / np.linalg.norm(reference)
+    assert error < 1e-10
+    assert list(tmp_path.iterdir()) == []
+
+
+def test_gpu_pool_limit_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
+    """The cap on cupy's pool is lifted after the sweep, also after an error."""
+    rng = np.random.default_rng(8)
+    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
+    pool = cupy.get_default_memory_pool()
+    previous = pool.get_limit()
+
+    src(*stack, chi_out=4, seed=0, device="gpu")
+    assert pool.get_limit() == previous
+
+    def boom(*_args: object) -> None:
+        msg = "boom"
+        raise RuntimeError(msg)
+
+    monkeypatch.setattr(SiteKernels, "sketch", boom)
+    with pytest.raises(RuntimeError, match="boom"):
+        src(*stack, chi_out=4, seed=0, device="gpu")
+    assert pool.get_limit() == previous
```

- [ ] **Step 2: Run them**

Run on a machine with CuPy and a GPU: `uv run --extra gpu-nvidia pytest tests/test_gpu_backend.py -o log_cli=false`
Expected: all pass. Without CuPy the module is skipped: `1 skipped`.

- [ ] **Step 3: Lint**

```bash
uv run ruff format tests/test_gpu_backend.py
uv run ruff check tests/test_gpu_backend.py
```
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
but diff
# Pick the IDs of exactly these files: tests/test_gpu_backend.py
but commit -b feat/src-out-of-core -m $'test(gpu): ✅ every environment tier and the pool cap on the GPU\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `tests/test_gpu_backend.py`; never the untracked `.codegraph/`.

---

### Task 10: Cluster acceptance benchmark

**Files:**
- Create: `benches/large/bench_large.py`, `benches/large/README.md`
- Modify: `benches/README.md`

**Interfaces:**
- Consumes: `src`, `Resources`, `make_plan` (spied on `src_method._sweep` to record the plan).
- Produces: the `generate`, `run` and `compare` commands.

`benches/` is excluded from the `ruff` hooks, like the other benchmarks; keep the file formatted anyway.

- [ ] **Step 1: Create the benchmark**

Create `benches/large/bench_large.py`:

```python
"""Out-of-core SRC of ``N . V . M . U`` with a large ``M``, read from disk.

Three commands:

- ``generate``: write the four MPOs site by site as ``.npy`` files, so that ``M``
  is never whole in memory.
- ``run``: compress the stack with automatic (or given) budgets and report the
  plan, the wall time, the device pool size, the host peak and the stall time.
- ``compare``: run twice with two GPU budgets and report the relative distance
  between the two outputs, which checks that the plan does not change the result.

The per-pass times and the stall time are also logged by `src_method` itself; set
``LOG_LEVEL_SRC=DEBUG`` to see them.
"""

from __future__ import annotations

import logging
import resource
from pathlib import Path  # noqa: TC003  (cyclopts reads the annotations at runtime)
from time import perf_counter

import cyclopts
import numpy as np
import structlog

import src_method._sweep as sweep_module
from src_method import Resources, src
from src_method._plan import make_plan

logger = structlog.get_logger()
app = cyclopts.App(help="Benchmark out-of-core SRC of N.V.M.U with a large M.")

LAYERS = ("N", "V", "M", "U")
PHYS = 4  # Pauli transfer matrix legs


def _site_shape(j: int, n_sites: int, bond: int) -> tuple[int, ...]:
    left = () if j == 0 else (bond,)
    right = () if j == n_sites - 1 else (bond,)
    return (*left, *right, PHYS, PHYS)


@app.command
def generate(
    directory: Path,
    *,
    n_sites: int = 50,
    bond: int = 4,
    bond_m: int = 4000,
    dtype: str = "complex128",
    seed: int = 0,
) -> None:
    """Write random MPOs ``N``, ``V``, ``M``, ``U`` as one ``.npy`` file per site.

    Args:
        directory: Where to write, ideally node-local NVMe.
        n_sites: Number of sites.
        bond: Bond dimension of ``N``, ``V`` and ``U``.
        bond_m: Bond dimension of ``M``.
        dtype: ``float64`` or ``complex128``.
        seed: Seed of the random draws.
    """
    rng = np.random.default_rng(seed)
    kind = np.dtype(dtype)
    for name in LAYERS:
        chi = bond_m if name == "M" else bond
        (directory / name).mkdir(parents=True, exist_ok=True)
        # Scaled so that products of the layers stay of order one.
        scale = 1 / np.sqrt(chi * PHYS)
        for j in range(n_sites):
            shape = _site_shape(j, n_sites, chi)
            site = np.lib.format.open_memmap(
                directory / name / f"{j:04d}.npy", mode="w+", dtype=kind, shape=shape
            )
            for row in range(shape[0]):  # one left-bond slice at a time
                draw = rng.normal(size=shape[1:])
                if kind.kind == "c":
                    draw = draw + 1j * rng.normal(size=shape[1:])
                site[row] = draw * scale
            site.flush()
            del site
        logger.info("Layer written", layer=name, bond=chi)


def _load(directory: Path) -> list[list[np.ndarray]]:
    return [
        [
            np.load(path, mmap_mode="r")
            for path in sorted((directory / name).glob("*.npy"))
        ]
        for name in LAYERS
    ]


def _run(
    directory: Path, chi_out: int, resources: Resources, seed: int
) -> list[np.ndarray]:
    layers = _load(directory)
    plans = []

    def spy(*args: object, **kwargs: object) -> object:
        plans.append(make_plan(*args, **kwargs))
        return plans[-1]

    sweep_module.make_plan = spy  # record the plan of the run
    try:
        start = perf_counter()
        out = src(
            *layers,
            chi_out=chi_out,
            dtype=layers[0][0].dtype,
            seed=seed,
            device="gpu",
            resources=resources,
        )
        seconds = perf_counter() - start
    finally:
        sweep_module.make_plan = make_plan
    import cupy  # noqa: PLC0415  (GPU-only benchmark)

    (plan,) = plans
    tiers = [site.tier for site in plan.sites]
    logger.info(
        "Run complete",
        seconds=round(seconds, 1),
        pool_bytes=cupy.get_default_memory_pool().total_bytes(),
        host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        planned_device_peak=plan.device_peak,
        planned_host_peak=plan.host_peak,
        disk_bytes=plan.disk_bytes,
        tiers={tier: tiers.count(tier) for tier in ("device", "host", "disk")},
        sketch_batches=sorted({site.sketch_batch for site in plan.sites[1:]}),
    )
    return out


@app.command
def run(
    directory: Path,
    *,
    chi_out: int = 2000,
    gpu_memory: str | None = None,
    host_memory: str | None = None,
    scratch_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Compress the stack written by ``generate`` on the GPU.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        gpu_memory: GPU budget, e.g. ``36GB``; detected when omitted.
        host_memory: Host budget; detected when omitted.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        seed: Seed of the sketch.
    """
    _run(directory, chi_out, Resources(gpu_memory, host_memory, scratch_dir), seed)


def _inner(a: list[np.ndarray], b: list[np.ndarray]) -> complex:
    """Frobenius inner product ``<a, b>`` of two MPOs, site by site."""
    env = np.einsum("rud,sud->rs", a[0].conj(), b[0])
    for x, y in zip(a[1:-1], b[1:-1]):
        env = np.einsum("rs,rtud,svud->tv", env, x.conj(), y, optimize=True)
    return complex(np.einsum("rs,rud,sud->", env, a[-1].conj(), b[-1]))


@app.command
def compare(
    directory: Path,
    *,
    chi_out: int = 500,
    small: str = "40GB",
    large: str = "80GB",
    scratch_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Run with two GPU budgets and report the relative distance of the outputs.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        small: The smaller GPU budget.
        large: The larger GPU budget.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        seed: Seed of the sketch, the same for both runs.
    """
    first = _run(directory, chi_out, Resources(small, None, scratch_dir), seed)
    second = _run(directory, chi_out, Resources(large, None, scratch_dir), seed)
    aa, bb, ab = _inner(first, first), _inner(second, second), _inner(first, second)
    distance = np.sqrt(max((aa + bb - 2 * ab).real, 0.0) / aa.real)
    logger.info("Relative distance between the runs", distance=distance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
```

- [ ] **Step 2: Smoke-test the generator and the inner product on the CPU**

Run:

```bash
uv run python - <<'EOF'
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path[:0] = ["benches/large"]
import bench_large as b
from src_method import src
d = Path(tempfile.mkdtemp())
b.generate(d, n_sites=4, bond=2, bond_m=3)
layers = b._load(d)
out = src(*layers, chi_out=8, dtype=np.complex128, seed=0)
print([t.shape for t in out], abs(b._inner(out, out)) > 0)
EOF
```

Expected: `[(8, 4, 4), (8, 8, 4, 4), (8, 8, 4, 4), (8, 4, 4)] True`.

- [ ] **Step 3: Document it**

Create `benches/large/README.md`:

````markdown
# Out-of-core benchmark

`bench_large.py` compresses the stack `N . V . M . U` of Pauli-transfer-matrix MPOs
(physical legs of 4) on one GPU, where `M` has a large bond and is read from disk.

```bash
# Write the stack to node-local NVMe (205 GB for M at the defaults, complex128).
uv run python benches/large/bench_large.py generate /local/stack --bond-m 4000

# Compress it with detected budgets, spilling to the same disk.
uv run python benches/large/bench_large.py run /local/stack --chi-out 2000 \
    --scratch-dir /local/scratch

# Check that the plan does not change the result: two GPU budgets, same seed.
uv run python benches/large/bench_large.py generate /local/medium --bond-m 1000
uv run python benches/large/bench_large.py compare /local/medium --chi-out 500 \
    --small 40GB --large 80GB --scratch-dir /local/scratch
```

`run` logs the wall time, the size of CuPy's pool at the end (its high-water mark,
since the pool keeps its blocks), the host peak (`ru_maxrss`), the planned peaks,
the bytes spilled and the tiers. With `LOG_LEVEL_SRC=DEBUG`, `src_method` also logs
the time of each pass, and at `info` the stall time (`SRC stalls`).

Acceptance of the out-of-core sweep on one A100-40GB, with at least 300 GB of host
memory and node-local NVMe:

1. `run` completes at 50 sites, `D_M = 4000`, `chi_out = 2000`, complex128.
2. The pool size stays within the GPU budget and the host peak within the host
   budget.
3. The stall time is below 10% of the wall time.
4. `compare` at `D_M = 1000`, `chi_out = 500` reports a distance below `1e-10`.
````

Save as `/tmp/t10_benches_readme.diff` and apply from the repository root with `git apply /tmp/t10_benches_readme.diff` (it only touches the working tree):

```diff
--- a/benches/README.md
+++ b/benches/README.md
@@ -10,3 +10,6 @@
 - [`stack/`](stack/) — One-shot SRC over stacks of trains against sequential
   pairwise application, by stack depth: accuracy against dense references and
   wall time. See [`stack/README.md`](stack/README.md).
+- [`large/`](large/) — Out-of-core SRC of `N . V . M . U` with a large `M` read
+  from disk, on one GPU: plan, wall time, memory peaks and stall time. See
+  [`large/README.md`](large/README.md).
```

- [ ] **Step 4: Format**

Run: `uv run ruff format benches/large`
Expected: `1 file left unchanged` (or reformatted once).

- [ ] **Step 5: Commit**

```bash
but diff
# Pick the IDs of exactly these files: benches/large/bench_large.py, benches/large/README.md, benches/README.md
but commit -b feat/src-out-of-core -m $'perf(bench): 📈 out-of-core acceptance benchmark for N.V.M.U\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `benches/large/bench_large.py`, `benches/large/README.md`, `benches/README.md`; never the untracked `.codegraph/`.

---

### Task 11: Documentation and final verification

**Files:**
- Create: `docs/large-problems.md`
- Modify: `mkdocs.yml`, `README.md`, `docs/developer-guide/testing.md`

**Interfaces:** none.

- [ ] **Step 1: Write the user page**

Create `docs/large-problems.md`:

````markdown
# Large problems

A single SRC sweep keeps, besides the input and output trains, one sketched
environment per site. For stacks with a large bond, such as `N . V . M . U` with a
bond of several thousand in `M`, these environments and the intermediates of the
contractions no longer fit on a GPU, and often not in host memory either.
`src_method` then plans the sweep to the memory at hand:

- the input cores are read one site at a time, so a train can live on disk;
- every contraction runs in batches of sketch columns (or rows), sized to the GPU
  budget;
- each environment is kept on the GPU, in host memory or in a scratch directory,
  the most recent ones on the fastest tier.

None of this changes the result beyond floating-point rounding: the random draws
and the mathematics are those of a sweep that fits in memory.

## Budgets

The budgets are set with `Resources`, passed to `src`, `apply` or `compress`:

```python
from src_method import Resources, src

out = src(
    N, V, M, U,
    chi_out=2000,
    dtype=np.complex128,
    device="gpu",
    resources=Resources(gpu_memory="36GB", scratch_dir="/local/scratch"),
)
```

Every field left unset is detected when the call starts:

| Field | Default |
|---|---|
| `gpu_memory` | free device memory, plus the free bytes of CuPy's pool, minus `max(10%, 1 GiB)` |
| `host_memory` | `MemAvailable` from `/proc/meminfo`, minus 10% |
| `scratch_dir` | `tempfile.gettempdir()`, which honours `$TMPDIR` |

Sizes are byte counts or strings: `"36GB"` is `36 * 10**9` bytes, `"36GiB"` is
`36 * 2**30`. On the CPU there is a single budget, `host_memory`. Host memory is
measured when the call starts, so inputs already held in memory are not counted
twice.

During the sweep, CuPy's default memory pool is capped at the GPU budget, so that
an estimate that falls short fails at once rather than when some other allocation
does; the previous limit is restored afterwards. If a single site does not fit the
budget even with batches of one column, `src` raises `MemoryError` naming the site.

## Inputs on disk

A train is any sequence of per-site array-likes with `shape`, `dtype` and
`np.asarray` support: NumPy arrays, `np.memmap`, zarr or HDF5 datasets. Each core
is read when the sweep reaches its site, once per pass, and a background thread
reads the next site while the current one is computed. Raw `.npy` files opened with
`np.load(path, mmap_mode="r")` are the fastest option; compressed formats may be
limited by decompression.

## The scratch directory

Environments that fit in neither budget are written to a per-process directory,
`<scratch_dir>/src-<pid>-<id>/`, one file per site. Use node-local disk: at the
reference size of 50 sites, `D_M = 4000` and `chi_out = 2000` in complex128, up to
410 GB are written and read back once. The directory is removed when the call
returns, also after an error; a process killed with `SIGKILL` leaves it behind, and
its name identifies the process.

## Reading the plan

Every sweep logs its plan at `info`:

- `tiers`: where the environment of each site lives (`device`, `host` or `disk`);
- `batches`: for each site, the batch of the environment, sketch and projection
  steps;
- `device_peak_bytes`, `host_peak_bytes`, `disk_bytes`: the planned peaks;
- `prefetch`: whether the next site is read ahead.

At the end, `SRC stalls` reports the seconds the sweep waited for input cores
(`site_seconds`) and for environments (`environment_seconds`). If they are a large
fraction of the run, the disk is too slow for the compute. Set
`LOG_LEVEL_SRC=DEBUG` to also get the time of each pass.
````

- [ ] **Step 2: Link it and document the test knobs**

Save as `/tmp/t11_mkdocs.diff` and apply from the repository root with `git apply /tmp/t11_mkdocs.diff` (it only touches the working tree):

```diff
--- a/mkdocs.yml
+++ b/mkdocs.yml
@@ -5,6 +5,7 @@
   - src/src_method
 nav:
   - Home: index.md
+  - 'Large Problems': large-problems.md
   - 'Developer Guide':
     - 'Versioning Scheme': developer-guide/versioning.md
     - 'Managing Dependencies': developer-guide/dependencies.md
```

Save as `/tmp/t11_readme.diff` and apply from the repository root with `git apply /tmp/t11_readme.diff` (it only touches the working tree):

```diff
--- a/README.md
+++ b/README.md
@@ -87,6 +87,19 @@

 At runtime, pass `device="gpu"` to use GPU acceleration. The library handles backend dispatch automatically.

+### Large Problems
+
+Stacks that do not fit on the GPU, or in host memory, still run: the sweep reads the input cores one site at a time (from NumPy arrays, `np.memmap`, zarr or HDF5 datasets), batches every contraction to a memory budget, and keeps the sketched environments on the GPU, in host memory or on local disk. The budgets are detected, or set explicitly with `Resources`:
+
+```python
+from src_method import Resources, src
+
+out = src(N, V, M, U, chi_out=2000, device="gpu",
+          resources=Resources(gpu_memory="36GB", scratch_dir="/local/scratch"))
+```
+
+See [Large problems](docs/large-problems.md) for the budgets, the scratch directory and how to read the logged plan.
+
 ## Installation

 ```bash
```

Save as `/tmp/t11_testing.diff` and apply from the repository root with `git apply /tmp/t11_testing.diff` (it only touches the working tree):

```diff
--- a/docs/developer-guide/testing.md
+++ b/docs/developer-guide/testing.md
@@ -253,6 +253,19 @@
     assert pytest.helpers.foo(True) is True
 ```

+### Exercising the out-of-core paths
+
+The batched contractions and the host and disk tiers of the sweep only engage when
+the budgets are tight, so tests force them with tiny budgets: on the CPU,
+`Resources(host_memory="1MB", scratch_dir=tmp_path)` gives small batches and spills
+the environments of a depth-4 stack with bonds of 4 to `tmp_path` (see
+`tests/test_stack.py`). Compare the dense operator of the result with that of a
+default run, not the cores: batching changes the rounding, and with it the cores of
+an ill-conditioned sketch, but not the operator they represent. The planner is a
+pure function, so `tests/test_plan.py` checks batch sizes and tiers from shapes
+alone, and `tests/test_gpu_backend.py` derives GPU budgets from a plan made with
+`make_plan` to reach every tier.
+
 ### How to use logging with tests

 Within `aurora`, we adopt a `pytest` configuration that allows to see the output
```

- [ ] **Step 3: Build the docs**

Run: `uv run mkdocs build`
Expected: the build succeeds and `site/large-problems/index.html` exists.

- [ ] **Step 4: Run every check**

Run:

```bash
uv run prek run --all-files
uv run pytest
```

Expected: every hook passes; every test passes (the GPU module skips without CuPy).

- [ ] **Step 5: Commit**

```bash
but diff
# Pick the IDs of exactly these files: docs/large-problems.md, mkdocs.yml, README.md, docs/developer-guide/testing.md
but commit -b feat/src-out-of-core -m $'docs: 📝 large problems, Resources and the out-of-core test knobs\n\nAssisted-by: Pi:claude-opus-5-5' <id> <id> ...
```

Commit only `docs/large-problems.md`, `mkdocs.yml`, `README.md`, `docs/developer-guide/testing.md`; never the untracked `.codegraph/`.

- [ ] **Step 6: Acceptance on the cluster**

On a node with one A100-40GB, at least 300 GB of host memory and node-local NVMe, follow `benches/large/README.md` and record the four acceptance criteria (completion, peaks within budgets, stall time below 10% of wall time, `compare` distance below `1e-10`) in `benches/large/README.md` under a `## Results` section, then commit that file with `perf(bench): 📈 out-of-core acceptance results`.
