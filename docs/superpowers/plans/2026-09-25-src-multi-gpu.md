# SRC Sweep over the GPUs of a Node (Phase 2): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one `src` sweep on the GPUs of a node, one process per GPU, with the sketch index split among them, so that the Phase 1 reference problem finishes close to `G` times faster on `G` GPUs.

**Architecture:** A small `Communicator` interface in `_comm.py` (single process, `mpi4py` on host arrays, NCCL on CuPy arrays) carries the few collectives the sweep needs. `src` builds the communicator from `Resources(comm=...)`, agrees on the call and the seed, and hands the output to a sink (rank 0 in memory, or `.npy` files and memmaps on every rank). The Phase 1 driver keeps one copy of the maths: each rank owns a cyclic subset of the sketch columns, and the right-to-left pass all-gathers the sketch columns and the rows of the projected environment at every site. The planner plans each rank's share.

**Tech Stack:** Python 3.11-3.14, NumPy and CuPy (through `src_method.utils._backend`), `opt_einsum`, `structlog`, `mpi4py>=4` (new `mpi` extra), NCCL through `cupy.cuda.nccl` (`nvidia-nccl-cu12` in the `gpu-nvidia` extra), `pytest`, `cyclopts` (benchmark), `uv`, `ruff` through `prek`, GitButler (`but`).

**Spec:** `docs/superpowers/specs/2026-09-25-src-multi-gpu-design.md` (read it first; this plan argues from it). Phase 1 context: `docs/superpowers/specs/2026-09-25-src-out-of-core-design.md`.

## How to read this plan

This plan was written under a rule of the design phase: no code in full. Every task
gives the files, the exact interfaces (signatures, types, error messages), the
behaviour in prose or pseudocode, and the tests to write, each with its setup and
its assertions. The executor writes the code, test first, and runs the commands
given. Where a detail is left to the executor, the plan says so.

## Global Constraints

- Work on branch `feat/src-multi-gpu`, stacked on `feat/src-out-of-core`; the spec is committed there. Commit with `but commit -b feat/src-multi-gpu -m ... <ids>` (IDs from `but diff`), never `git add`/`git commit`, and never the untracked `.codegraph/`.
- `from __future__ import annotations` at the top of every module under `src/`; type hints everywhere; Google-style docstrings without types on every public function, class and module; `ruff` is the source of truth for style.
- Backend-specific calls (CuPy, NCCL, device selection) live in `src_method.utils._backend` or in `_comm.py`; the algorithms get `xp` and a `Communicator`. `mpi4py` and `cupy.cuda.nccl` are imported lazily, only when a communicator is given: a single-process run imports neither.
- Log with `structlog`, never `print`, and never `warnings.warn` for run-time advice (`filterwarnings = ["error"]` turns warnings into test failures).
- `src`, `apply`, `compress` stay pure: never mutate inputs.
- Commits: `<type>(<optional scope>): <gitmoji> <description>`, ending with `Assisted-by: Pi:claude-opus-5-5`, no `Co-authored-by`.
- Before pushing: `uv run prek run --all-files` and `uv run pytest` pass.
- Decisions of the spec, verbatim: one process per GPU (SPMD) launched with `srun` or `mpirun`; NCCL through `cupy.cuda.nccl` with `mpi4py` for bootstrap; `Resources(comm=...)`, nothing collective without it; output on rank 0 in memory, or `.npy` files and memmaps on every rank with `Resources(output_dir=...)`; large inputs file-backed; success is strong scaling (at least 80% parallel efficiency at 8 GPUs on 8xA100 and 8xH100 at the reference size, at least 90% on 2xA100).
- Without a communicator, or with one of size one, the sweep is exactly Phase 1: the existing tests must pass unchanged.

## Refinements of the spec

Settled while planning; the executor should not undo them.

1. **`Communicator.abort(code)`** joins the interface: the driver calls it when an exception escapes during the sweep on more than one rank. `SingleComm.abort` is never called.
2. **The truncation rank is agreed through a callback**: `truncated_qr(matrix, cutoff, xp, agree_rank=None)` computes its local rank from the singular values of `R` and passes it through `agree_rank` (rank 0's value, broadcast) before truncating. The spec's "optional `rank` argument" cannot work, because the rank is only known inside the function.
3. **Output sinks and `unpad_site`**: the output goes through a small sink object, and each core is unpadded as it is produced, with a new `unpad_site` that mirrors `pad_site`.
4. **Large in-memory inputs are logged, not warned about**: when a distributed call on more than one rank per node gets an in-memory `np.ndarray` layer (not a memmap) above 1 GiB, `src` logs a warning through `structlog` that every rank holds a copy.
5. **CI uses the system MPICH** on the Ubuntu runner (`apt-get install mpich`) with the `mpi4py` wheel from PyPI; the MPI tests skip wherever `mpiexec` or a working `mpi4py` is missing.

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/src_method/_comm.py` | create | `Communicator`, `SingleComm`, `MpiHostComm`, `NcclComm`, `make_communicator`; block and share helpers; `agree`, `collectively` |
| `src/src_method/_output.py` | create | output sinks: memory, directory, none |
| `src/src_method/_tensor_train.py` | modify | `unpad_site` |
| `src/src_method/utils/linalg.py` | modify | `truncated_qr(..., agree_rank=None)` |
| `src/src_method/utils/_backend.py` | modify | `select_device(local_rank, xp)`, `nccl_module(xp)` |
| `src/src_method/_plan.py` | modify | `Resources.comm`, `Resources.output_dir`; per-rank shares in `resolve_budgets`; `make_plan(..., columns=, rows=, gather=, holds_output=)` |
| `src/src_method/_sweep.py` | modify | column ownership, the two all-gathers, rank-0 output, collective error handling |
| `src/src_method/stack.py` | modify | communicator, fingerprint, seed, device, sinks |
| `pyproject.toml`, `uv.lock`, `flake.nix`, `.github/workflows/test.yml` | modify | `mpi` extra, NCCL wheel, MPICH in the dev shell and in CI |
| `tests/conftest.py` | create | `mpirun` fixture |
| `tests/mpi_scripts/*.py` | create | scripts run under `mpiexec` |
| `tests/test_comm.py`, `tests/test_output.py` | create | unit and MPI tests |
| `tests/test_plan.py`, `tests/test_stack.py`, `tests/test_tensor_train.py`, `tests/test_backend.py`, `tests/test_gpu_backend.py`, `tests/test_linalg.py` | modify / create | planner, integration, helpers, GPU |
| `benches/large/bench_large.py`, `benches/large/README.md` | modify | MPI mode, `scaling` and `efficiency` commands |
| `docs/large-problems.md`, `README.md`, `docs/developer-guide/dependencies.md`, `docs/developer-guide/testing.md` | modify | documentation |

---

### Task 1: Work splitting and block helpers, and the single-process communicator

**Files:**
- Create: `src/src_method/_comm.py`
- Test: `tests/test_comm.py` (create)

**Interfaces (produced):**

```python
class Communicator(Protocol):
    rank: int
    size: int
    local_rank: int
    local_size: int
    def allgather(self, local: NDArray, axis: int) -> NDArray: ...
    def allgather_objects(self, value: object) -> list[object]: ...
    def bcast_int(self, value: int, root: int = 0) -> int: ...
    def barrier(self) -> None: ...
    def abort(self, code: int) -> None: ...

class SingleComm:            # rank 0 of 1, local rank 0 of 1
    ...                      # allgather returns `local` itself; allgather_objects -> [value];
                             # bcast_int -> value; barrier no-op; abort raises SystemExit(code)

def owned_columns(n: int, rank: int, size: int) -> slice    # slice(rank, n, size)
def row_block(n: int, rank: int, size: int) -> slice        # contiguous, sizes differ by <= 1,
                                                            # lower ranks take the larger blocks
def pad_blocks(block: NDArray, axis: int, rows: int) -> NDArray
    # move `axis` to the front, make contiguous, pad with zeros to `rows` along it
def join_blocks(gathered: NDArray, counts: Sequence[int], axis: int) -> NDArray
    # gathered: (size, rows_max, *rest), padded blocks in rank order;
    # take counts[r] rows of block r, concatenate, move the axis back
def agree(comm: Communicator, fn: Callable[[], T]) -> T
def collectively(comm: Communicator, fn: Callable[[], T]) -> T
```

Behaviour:

- `agree` runs `fn` on every rank, then all-gathers `None` or `(type name, message)` of the exception it raised. If any rank failed, every rank raises an exception of the same type (one of `ValueError`, `MemoryError`, `TypeError`, otherwise `RuntimeError`) with the message of the lowest failing rank, prefixed `"Rank {r}: "`. Without failures it returns `fn()`'s result. With `SingleComm` it is `fn()`.
- `collectively` runs `fn`; if an exception escapes and `comm.size > 1`, it logs it with `logger.exception("SRC failed; aborting", rank=comm.rank)` and calls `comm.abort(1)`; with one rank it re-raises.

Tests to write (`tests/test_comm.py`):

- [ ] **Step 1: Write the failing tests**
  - `owned_columns`: for `n in (1, 7, 8, 2000)` and `size in (1, 2, 3, 8)`, the ranges of all ranks partition `range(n)` exactly once; rank `r` gets `ceil((n - r) / size)` columns.
  - `row_block`: same partition property; blocks are contiguous and in rank order; block sizes differ by at most one.
  - `pad_blocks` then `join_blocks` round-trips random blocks of uneven sizes along axis 0 and axis 3 of a rank-4 array, float64 and complex128: joining the padded blocks of three ranks equals `np.concatenate(blocks, axis)`.
  - `SingleComm`: `allgather(x, axis)` is `x`; `allgather_objects(v) == [v]`; `bcast_int(5) == 5`; `rank, size, local_rank, local_size == 0, 1, 0, 1`.
  - `agree` with `SingleComm`: returns the value; re-raises a `MemoryError` from `fn` unchanged in type.
  - `collectively` with `SingleComm`: re-raises.
- [ ] **Step 2: Run** `uv run pytest tests/test_comm.py -o log_cli=false`; expected: `ModuleNotFoundError: No module named 'src_method._comm'`.
- [ ] **Step 3: Implement** `_comm.py` with the interfaces above. Keep `mpi4py` and CuPy imports out of this task.
- [ ] **Step 4: Run** the tests; expected: all pass.
- [ ] **Step 5: Lint** `uv run ruff format src/src_method/_comm.py tests/test_comm.py && uv run ruff check src tests`.
- [ ] **Step 6: Commit** `feat(comm): ✨ work splitting, block helpers and the single-process communicator`.

---

### Task 2: MPI on host arrays, the `mpi` extra and the MPI test harness

**Files:**
- Modify: `src/src_method/_comm.py` (add `MpiHostComm`, `make_communicator`)
- Modify: `pyproject.toml` (extra `mpi = ["mpi4py>=4"]`), `uv.lock` (`uv lock`), `flake.nix` (add `pkgs.mpich` to the dev shell packages and its `lib` to `LD_LIBRARY_PATH`), `.github/workflows/test.yml` (Ubuntu runner: `sudo apt-get install -y mpich` before syncing, and `--extra mpi` in `uv sync`; the GPU runner is unchanged)
- Create: `tests/conftest.py`, `tests/mpi_scripts/collectives.py`
- Test: `tests/test_comm.py` (append)

**Interfaces (produced):**

```python
class MpiHostComm:
    def __init__(self, comm: Any) -> None: ...   # an mpi4py Comm
    # rank/size from comm; local_rank/local_size from comm.Split_type(MPI.COMM_TYPE_SHARED)
    # allgather: moveaxis + contiguous uint8 view, Allgatherv with byte counts and
    #            displacements from an allgather of the block lengths, then join_blocks
    # allgather_objects: comm.allgather; bcast_int: comm.bcast; barrier: comm.Barrier
    # abort: comm.Abort(code)

def make_communicator(comm: Any | None, xp: ModuleType) -> Communicator:
    # None or comm.Get_size() == 1 -> SingleComm()
    # host backend -> MpiHostComm(comm); GPU backend -> NcclComm(comm, xp) (Task 3)
```

Test harness (`tests/conftest.py`):

```python
@pytest.fixture
def mpirun(tmp_path) -> Callable[..., MpiResult]:
    # skip if shutil.which("mpiexec") is None, or if
    #   `python -c "from mpi4py import MPI"` fails in a subprocess
    # run(script: str, n: int, *args: str, timeout: float = 120) -> MpiResult
    #   command: mpiexec [--oversubscribe if Open MPI] -n n sys.executable
    #            tests/mpi_scripts/<script> <tmp_path> *args
    #   env: LOG_LEVEL_SRC=INFO, OMP_NUM_THREADS=1
    # MpiResult: returncode, output (stdout + stderr), and per-rank results loaded from
    #   <tmp_path>/rank-<r>.npz (a dict of arrays; missing file -> None)
```

A launch that exceeds `timeout` fails the test with the captured output (a hang is a failure, not a stall). Scripts in `tests/mpi_scripts/` take the result directory as their first argument and write `rank-<r>.npz`; they are not collected by `pytest` (no `test_` prefix) and need type annotations (`ANN` applies to them).

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - `tests/mpi_scripts/collectives.py` builds on every rank a block of `rank + 1` rows along axis 1 of a `(2, rank + 1, 3)` complex array with entries `rank * 100 + arange`, all-gathers along axis 1, all-gathers the object `(rank, "x")`, broadcasts `rank + 41` from rank 0, and saves the results plus `local_rank` and `local_size`.
  - `test_mpi_host_collectives(mpirun)`: with `n = 3`, every rank's gathered array equals the concatenation of the three expected blocks; the objects are `[(0, "x"), (1, "x"), (2, "x")]`; the broadcast is 41 everywhere; local ranks are `0, 1, 2` and local sizes 3.
  - `test_make_communicator_single_process`: `make_communicator(None, np)` is a `SingleComm`.
- [ ] **Step 2: Run** `uv run pytest tests/test_comm.py -o log_cli=false`; expected: the MPI test fails on the missing `MpiHostComm` (or skips if MPI is absent: install MPICH and `uv sync --extra mpi` first).
- [ ] **Step 3: Implement** `MpiHostComm`, `make_communicator`, the fixture and the script; add the extra, lock, update the flake and CI.
- [ ] **Step 4: Run** the tests with MPI available; expected: pass. Check that `uv run pytest` without the `mpi` extra skips the MPI test instead of failing.
- [ ] **Step 5: Lint** as in Task 1, including `tests/conftest.py` and `tests/mpi_scripts`.
- [ ] **Step 6: Commit** `feat(comm): ✨ MPI collectives on host arrays and an MPI test harness` (files: `_comm.py`, `pyproject.toml`, `uv.lock`, `flake.nix`, `.github/workflows/test.yml`, `tests/conftest.py`, `tests/mpi_scripts/collectives.py`, `tests/test_comm.py`).

---

### Task 3: NCCL on device arrays

**Files:**
- Modify: `src/src_method/_comm.py` (`NcclComm`), `src/src_method/utils/_backend.py` (`nccl_module`, `select_device`), `pyproject.toml` (`nvidia-nccl-cu12` in `gpu-nvidia`), `uv.lock`
- Test: `tests/test_backend.py` (append), `tests/test_gpu_backend.py` (append), `tests/mpi_scripts/collectives.py` (a `--device gpu` option)

**Interfaces (produced):**

```python
# utils/_backend.py
def nccl_module(xp: ModuleType) -> ModuleType    # cupy.cuda.nccl; ImportError with an
                                                 # install hint if not nccl.available
def select_device(local_rank: int, xp: ModuleType) -> None
    # host backend: no-op; one visible device: no-op; n > 1: xp.cuda.Device(local_rank % n).use()

# _comm.py
class NcclComm:
    def __init__(self, comm: Any, xp: ModuleType) -> None: ...
    # host side as MpiHostComm (allgather_objects, bcast_int, barrier, abort, local ranks)
    # NCCL communicator: uid = nccl.get_unique_id() on rank 0 (bytes in CuPy 14, a tuple
    #   in CuPy 13: broadcast it as an object either way), then
    #   nccl.NcclCommunicator(size, uid, rank)
    # allgather(local, axis):
    #   counts = allgather_objects(local.shape[axis]); rows = max(counts)
    #   send = pad_blocks(local, axis, rows) viewed as uint8
    #   recv = xp.empty(size * send.nbytes, uint8)
    #   comm.allGather(send.data.ptr, recv.data.ptr, send.nbytes, nccl.NCCL_UINT8,
    #                  xp.cuda.get_current_stream().ptr)
    #   return join_blocks(recv.view(local.dtype).reshape(size, rows, *rest), counts, axis)
```

Collectives run on the current (compute) stream, so they are ordered with the kernels without events.

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - `tests/test_backend.py`: `select_device(3, np)` is a no-op; `nccl_module(np)` raises `ImportError` (the host backend has no NCCL).
  - `tests/test_gpu_backend.py`, skipped unless CuPy sees at least two devices and `mpiexec` works: `test_nccl_collectives(mpirun)` runs `collectives.py --device gpu` with `n = 2` and asserts what Task 2 asserts for the host version, now with CuPy arrays converted to NumPy before saving.
- [ ] **Step 2: Run** the host tests; expected: failure on the missing helpers.
- [ ] **Step 3: Implement** `nccl_module`, `select_device`, `NcclComm`, and route `make_communicator` to it on the GPU backend.
- [ ] **Step 4: Run** `uv run pytest tests/test_backend.py tests/test_comm.py -o log_cli=false`; expected: pass. The GPU test runs only on a multi-GPU machine: `uv run --extra gpu-nvidia --extra mpi pytest tests/test_gpu_backend.py -k nccl`.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(comm): ✨ NCCL all-gathers on device arrays`.

---

### Task 4: `Resources` fields and output sinks

**Files:**
- Modify: `src/src_method/_plan.py` (`Resources.comm`, `Resources.output_dir`), `src/src_method/_tensor_train.py` (`unpad_site`)
- Create: `src/src_method/_output.py`
- Test: `tests/test_output.py` (create), `tests/test_tensor_train.py` (append), `tests/test_plan.py` (append)

**Interfaces (produced):**

```python
# _plan.py
@dataclass(frozen=True)
class Resources:
    gpu_memory: int | str | None = None
    host_memory: int | str | None = None
    scratch_dir: str | os.PathLike[str] | None = None
    comm: Any = None                                   # an mpi4py communicator
    output_dir: str | os.PathLike[str] | None = None

# _tensor_train.py
def unpad_site(site: NDArray, kind: TrainKind, i: int, last: int) -> NDArray  # inverse of pad_site

# _output.py
class OutputSink(Protocol):
    def add(self, j: int, core: NDArray) -> None: ...          # core: unpadded, on the host
    def finish(self) -> list[NDArray] | None: ...

def make_sink(n_sites: int, rank: int, output_dir: Path | None,
              barrier: Callable[[], None]) -> OutputSink
    # no output_dir: rank 0 -> MemorySink (keeps the cores), others -> NullSink (None)
    # output_dir: rank 0 writes <output_dir>/site-<j:04d>.npy in add (np.save; the
    #   directory is created, existing site files are overwritten); finish() calls
    #   barrier() on every rank, then every rank returns
    #   [np.load(path, mmap_mode="r") for each site]
```

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - `unpad_site` inverts `pad_site` for every kind and position (compare with `unpad` on the whole train).
  - `Resources(comm=object(), output_dir="x")` constructs; the existing validation of the budgets is unchanged.
  - `MemorySink` returns the cores in site order whatever the order of `add` (the sweep adds from the last site down); `NullSink.finish()` is `None`.
  - Directory sink with a no-op barrier: after `add` of three cores in reverse order, `finish()` returns memmaps equal to the cores, and `site-0000.npy` .. `site-0002.npy` exist; a non-zero rank's sink writes nothing and still returns the memmaps once the files exist.
- [ ] **Step 2: Run**; expected: import errors.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run**; expected: pass.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(output): ✨ output sinks: rank 0 in memory or .npy files for every rank`.

---

### Task 5: Agreeing on the truncation rank

**Files:**
- Modify: `src/src_method/utils/linalg.py`
- Test: `tests/test_linalg.py` (create)

**Interface:** `truncated_qr(matrix, cutoff, xp=np, agree_rank: Callable[[int], int] | None = None) -> NDArray`. With `cutoff > 0`, the local rank `max(1, #(s >= cutoff * s_max))` is passed through `agree_rank` when given, and the isometry is truncated to the returned rank. With `cutoff <= 0` the callback is not called.

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - With no callback, results are unchanged from today (compare with a copy of the current behaviour on a random matrix with a clear spectral gap).
  - A callback that returns `local - 1` truncates one column more, and the columns kept are orthonormal.
  - A callback that records its argument sees the local rank; with `cutoff = 0` it is never called.
  - Wide matrices (`m < n`) follow the same rules.
- [ ] **Step 2: Run**; expected: `TypeError` on the unknown keyword.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `uv run pytest tests/test_linalg.py tests/test_package.py -o log_cli=false -m "not perf"`; expected: pass.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(linalg): ✨ let the caller agree on the truncation rank`.

---

### Task 6: Planning each rank's share

**Files:**
- Modify: `src/src_method/_plan.py` (`resolve_budgets`, `make_plan`)
- Test: `tests/test_plan.py` (append)

**Interfaces:**

```python
def resolve_budgets(resources: Resources | None, xp: ModuleType,
                    local_size: int = 1) -> Budgets
    # host and disk budgets divided by local_size (detected or explicit alike: an
    # explicit host_memory is the node's); device budget unchanged

def make_plan(site_shapes, site_bytes, chi_out, dtype, budgets, *,
              columns: int | None = None,     # this rank's sketch columns; default chi_out
              rows: int | None = None,        # this rank's projection rows; default chi_out
              gather: bool = False,           # add the two gather buffers
              holds_output: bool = True) -> Plan
```

Behaviour: environments are sized for `columns`; environment and sketch batches are limited by `columns`, projection batches by `rows`; with `gather`, the fixed memory of the sketch step gains the gathered sketch (`eta * p * chi_out`) plus the padded receive buffer (the same again), and that of the projection step the gathered new `S` (`chi_out * A`) plus its receive buffer; the output train counts against the host budget only when `holds_output`. With the defaults the plan is exactly Phase 1's.

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - Defaults reproduce the Phase 1 plan for the existing test shapes (equality of `Plan`).
  - `columns = chi // 4` gives environments a quarter of the size: the total planned environment bytes (sum over tiers) drop by 4; environment and sketch batches never exceed `columns`; projection batches never exceed `rows`.
  - `gather=True` raises `device_peak` by at least the two buffers of the widest site.
  - `holds_output=False` lowers `host_peak` by the output train.
  - `resolve_budgets(Resources(host_memory="8GB", scratch_dir=tmp_path), np, local_size=4).host == 2 * 10**9`, and the disk budget is a quarter of the free space.
- [ ] **Step 2: Run**; expected: `TypeError` on the new keywords.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `uv run pytest tests/test_plan.py -o log_cli=false`; expected: pass.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(plan): ✨ plan one rank's share of columns, rows and node memory`.

---

### Task 7: Preparing a distributed call in `src`

**Files:**
- Modify: `src/src_method/stack.py`
- Create: `tests/mpi_scripts/prepare.py`
- Test: `tests/test_stack.py` (append)

Behaviour of `src`, in order (pseudocode):

```text
comm = make_communicator(resources.comm if resources else None, xp)
select_device(comm.local_rank, xp)
agree(comm, check_fingerprint)        # all-gather the fingerprint; mismatch -> ValueError
    fingerprint = (tuple of (shape, dtype.str) of every site of every train, in the
                   original order), chi_out, cutoff, np.dtype(dtype).str,
                   seed (None allowed, but then None on every rank)
    message: "Ranks disagree on the call: rank {r} has {field} = {value}, rank 0 {value0}."
if seed is None and comm.size > 1: seed = comm.bcast_int(fresh 63-bit seed on rank 0)
prng = default_rng(seed)
warn_in_memory_inputs(layers, comm)   # structlog warning, see Refinement 4
sink = make_sink(n_sites, comm.rank, output_dir, comm.barrier)
exact path: cores computed on every rank; rank 0 adds them to the sink
sweep path: until Task 8, call sweep as today and let rank 0 add the returned cores
            to the sink; Task 8 replaces this with sweep(..., comm=comm, sink=sink)
return sink.finish()
```

With `comm=None` and no `output_dir`, `src` returns exactly what it returns today; with `output_dir` and no communicator, the cores are written and returned as memmaps.

Tests to write:

- [ ] **Step 1: Write the failing tests**
  - Single process: `src(..., resources=Resources(output_dir=tmp_path))` returns memmaps equal to `src(...)` for the same seed, and the files exist; same for a two-site stack (exact path).
  - `tests/mpi_scripts/prepare.py` runs `src` on a small fixed two-site stack (the exact path, which does not need Task 8) with `comm=MPI.COMM_WORLD`; option `--mismatch` makes rank 1 pass `chi_out + 1`. It saves either the result (on rank 0), a flag for `None` (other ranks), or the exception type and message.
  - `test_distributed_fingerprint_mismatch(mpirun)`: `n = 2`, `--mismatch`: exit status 0, and both ranks saved a `ValueError` whose message names rank 1 and `chi_out`.
  - `test_distributed_exact_path_output(mpirun)`: `n = 2`: rank 0's cores equal a single-process `src` on the same stack; rank 1 returns `None`.
- [ ] **Step 2: Run**; expected: failures on the missing behaviour.
- [ ] **Step 3: Implement** in `stack.py`, reusing `agree`, `make_communicator`, `select_device`, `make_sink`.
- [ ] **Step 4: Run** `uv run pytest tests/test_stack.py -o log_cli=false -m "not perf"`; expected: pass.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(stack): ✨ agree on the call, the seed and the output across ranks`.

---

### Task 8: The distributed driver

**Files:**
- Modify: `src/src_method/_sweep.py`
- Create: `tests/mpi_scripts/distributed_src.py`
- Test: `tests/test_stack.py` (append)

**Interface:** `sweep(layers, kind, chi_out, prng, xp, *, cutoff=0.0, dtype=np.float64, resources=None, comm: Communicator | None = None, sink: OutputSink | None = None) -> list[NDArray] | None`. With `comm=None` it behaves as today (a `SingleComm` and a memory sink internally) and returns the cores.

Behaviour (pseudocode, on top of the Phase 1 driver):

```text
cols = owned_columns(chi_out, comm.rank, comm.size); n_local = len(range(chi_out)[cols])
budgets = resolve_budgets(resources, xp, comm.local_size)
plan = agree(comm, lambda: make_plan(shapes, sizes, chi_out, work, budgets,
                                     columns=n_local,
                                     rows=len(range(chi_out)[row_block(chi_out, 0, comm.size)]),
                                     gather=comm.size > 1,
                                     holds_output=comm.rank == 0 and output kept in memory))
env_shapes use n_local instead of chi_out
collectively(comm, passes):
  left-to-right: omega = draw full (chi_out, up, down) as today, then omega[cols];
                 first environment ones((n_local, 1, ...)); batches over n_local
  right-to-left, per site j:
    local = empty((eta, up, down, n_local)); fill from batches
    sketch = comm.allgather(local, axis=3)                    # (eta, up, down, chi_out)
    Q = truncated_qr(sketch.reshape(rows, chi_out), cutoff, xp,
                     agree_rank=(lambda r: comm.bcast_int(r)) if comm.size > 1 else None)
    rows_j = row_block(Q.shape[1], comm.rank, comm.size)
    local_S = project rows rows_j of eta_j in batches (Phase 1 kernel)
    S = comm.allgather(local_S, axis=0)
    if comm.rank == 0: sink.add(j, unpad_site(to_numpy(eta_j), kind, j, last))
  first site: rank 0 only, added to the sink as site 0
log "SRC stalls" per rank (add rank=comm.rank to the log lines when comm.size > 1)
return sink.finish() only when the sink was created here; otherwise src finishes it
```

Tests to write (`tests/mpi_scripts/distributed_src.py` builds, on every rank, the same stack from a fixed seed: four complex MPOs with bonds `[3, 4, 4, 4, 3]`, 6 sites; options `--chi`, `--cutoff`, `--no-seed`, `--output-dir`, `--host-memory`, `--scratch-dir`, `--fail-on-rank R`, `--device`):

- [ ] **Step 1: Write the failing tests**
  - `test_distributed_matches_single(mpirun, n, chi)` for `(n, chi)` in `(2, 16), (3, 16), (3, 17)`: rank 0's cores give the same dense operator as a single-process `src` with the same seed, to `1e-10` relative Frobenius norm; the other ranks return `None`.
  - With `--cutoff 0.5`: same comparison, and rank 0's bond dimensions equal the single-process ones.
  - With `--output-dir`: every rank returns memmaps, identical across ranks, equal to rank 0's in-memory result of the previous test.
  - With `--host-memory 1MB --scratch-dir <tmp>`: the result still matches, and every rank's scratch directory is gone afterwards (the tmp directory is empty).
  - Planning failure: rank 1 gets `--host-memory 1kB`: every rank saves a `MemoryError` naming rank 1; exit status 0.
  - Abort: `--fail-on-rank 1` monkeypatches `SiteKernels.sketch` to raise on rank 1: nonzero exit status, and the output contains `aborting` and `rank=1`.
  - Seed agreement: `--no-seed` with `n = 2`: rank 0 logs the agreed seed (at `debug`, `seed=...`); a single-process `src` with that seed gives the same dense operator as rank 0's result.
  - The existing single-process tests pass unchanged (`uv run pytest -m "not perf"`).
- [ ] **Step 2: Run**; expected: failures (keyword `comm` unknown to `sweep`).
- [ ] **Step 3: Implement** in `_sweep.py`; keep one copy of the passes, with the communicator calls as the only difference.
- [ ] **Step 4: Run** `uv run pytest -o log_cli=false` with MPI available; expected: all pass, the two `perf` benchmarks within noise of Phase 1.
- [ ] **Step 5: Lint**.
- [ ] **Step 6: Commit** `feat(sweep): ✨ split the sketch index among ranks`.

---

### Task 9: Multi-GPU parity

**Files:**
- Test: `tests/test_gpu_backend.py` (append)

- [ ] **Step 1: Write the test** `test_nccl_sweep_matches_single_gpu(mpirun)`, skipped unless CuPy sees two devices and `mpiexec` works: run `distributed_src.py --device gpu --chi 17` with `n = 2` (an odd `chi` exercises the NCCL padding) and compare rank 0's dense operator with a single-GPU run in the test process, to `1e-10`.
- [ ] **Step 2: Run** on a multi-GPU node: `uv run --extra gpu-nvidia --extra mpi pytest tests/test_gpu_backend.py -o log_cli=false`; expected: pass. Elsewhere: skipped.
- [ ] **Step 3: Lint** and **commit** `test(gpu): ✅ NCCL sweep parity with a single GPU`.

---

### Task 10: Scaling benchmark

**Files:**
- Modify: `benches/large/bench_large.py`, `benches/large/README.md`

Behaviour:

- `run` and `compare` accept `--mpi`: `Resources(comm=MPI.COMM_WORLD, ...)`; every rank logs its plan, time, pool size and stall time with its rank.
- `scaling DIRECTORY --output-dir OUT --results FILE.jsonl`: run once under the current launch (`srun -n G`), write the output to `OUT/g<G>/`, and append one JSON line on rank 0: `{"gpus": G, "seconds": ..., "chi_out": ..., "bond_m": ...}`.
- `efficiency FILE.jsonl --output-dir OUT`: for each line, the parallel efficiency `T_1 / (G T_G)` against the `gpus = 1` line, and the relative distance of `OUT/g<G>/` to `OUT/g1/` from TT inner products (reuse `_inner`); log a table and fail (exit status 1) if a distance exceeds `1e-10`.
- README: how to launch the 1/2/4/8 series with `srun`, and the Phase 2 pass criteria.

- [ ] **Step 1: Implement** the commands.
- [ ] **Step 2: Smoke-test on the CPU:** with MPICH, `mpiexec -n 2 uv run python benches/large/bench_large.py scaling ... --device cpu` on a tiny generated stack (add a `--device` option defaulting to `gpu`), then `efficiency` on the two lines of a 1-rank and a 2-rank run; expected: efficiency printed, distance below `1e-10`.
- [ ] **Step 3: Format** and **commit** `perf(bench): 📈 multi-GPU scaling and efficiency commands`.

---

### Task 11: Documentation and final verification

**Files:**
- Modify: `docs/large-problems.md` (section "Several GPUs"), `README.md` (a sentence and a link under "Large Problems"), `docs/developer-guide/dependencies.md` (the `mpi` extra and an MPI library: system, or `mpich`/`openmpi` wheels from `https://pypi.anaconda.org/mpi4py/simple`, or `impi-rt`), `docs/developer-guide/testing.md` (running the MPI tests: install MPICH, `uv sync --extra mpi`, then `uv run pytest`; the `mpirun` fixture; timeouts)

Content of "Several GPUs": launching with `srun -n G` or `mpirun -n G`; `Resources(comm=MPI.COMM_WORLD, output_dir=...)`; every rank calls `src` with the same arguments; return values on rank 0 and the others; large inputs file-backed, and the logged warning otherwise; how the ranks of a node share host memory and scratch disk; an error during the sweep aborts the job; the expected efficiency and when it drops (small problems).

- [ ] **Step 1: Write** the documentation.
- [ ] **Step 2: Build** `uv run mkdocs build`; expected: success.
- [ ] **Step 3: Verify** `uv run prek run --all-files` and `uv run pytest` (with and without the `mpi` extra); expected: all pass, MPI tests skipped without it.
- [ ] **Step 4: Commit** `docs: 📝 running SRC on several GPUs`.
- [ ] **Step 5: Acceptance on the cluster:** the 1/2/4/8 series on 8xA100 and 8xH100 and the 1/2 series on 2xA100 at the reference size, with `efficiency` over each; record the tables and the pass criteria in `benches/large/README.md` under "Results (Phase 2)", and commit `perf(bench): 📈 multi-GPU scaling results`.

---

## Self-review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Decisions: SPMD, NCCL with `mpi4py` bootstrap | 2, 3 |
| Public interface: `comm`, `output_dir`, return values | 4, 7 |
| Contract: fingerprint, seed, device selection, exact path | 7 |
| Communicator: interface and three implementations | 1, 2, 3 |
| Driver: column ownership, two all-gathers, rank-0 output, first site | 8 |
| Truncation rank decided on rank 0 | 5, 8 |
| Planner: columns, rows, gather buffers, host and disk shares | 6, 8 |
| Errors: agreement before the sweep, abort during it | 1, 7, 8 |
| Dependencies: `mpi` extra, NCCL wheel | 2, 3 |
| Testing: unit, MPI on the CPU, GPU | 1-9 |
| Acceptance: scaling and efficiency, criteria | 10, 11 |
| Documentation | 11 |
| Future extensions (GPUDirect Storage, compression) | recorded in the spec; no task by design |

**Placeholders:** none; the details left to the executor are named as such (implementation bodies).

**Consistency:** `owned_columns`, `row_block`, `pad_blocks`, `join_blocks`, `agree`, `collectively`, `make_communicator`, `make_sink`, `unpad_site`, `select_device`, `nccl_module`, `truncated_qr(..., agree_rank=)`, `resolve_budgets(..., local_size=)` and `make_plan(..., columns=, rows=, gather=, holds_output=)` are defined once and used with the same names and types in later tasks.
