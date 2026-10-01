# Design: out-of-core SRC sweep on one GPU (Phase 1)

- Status: approved; refined while prototyping the implementation plan
- Follow-ups: Phase 2 (sketch-index split over the GPUs of a node) and Phase 3
  (bond split across nodes) build on this kernel; see
  [Forward compatibility](#forward-compatibility-with-phases-2-and-3).

## Goal

Make one `src` sweep run when the stack does not fit on the GPU. The target is the
contraction and compression of MPO stacks such as `N . V . M . U`, where only `M` is
large and has been produced by an earlier computation. The sweep must stream the
input cores, bound its working set by batching, and keep the environments on the
GPU, in host memory or on local disk, as the budgets allow. The mathematics of the
sweep does not change.

## Reference problem

| Quantity | Value |
|---|---|
| Sites | 50 |
| Physical legs of every MPO | 4 (Pauli transfer matrices), so `p = 16` output legs per site |
| `D_N`, `D_V`, `D_U` | 4 |
| `D_M` | 4000, provided by the user |
| `chi_out` (`l` below) | 2000 |
| dtype | complex128 |
| Hardware | one A100-40GB, at least 300 GB of host memory, node-local NVMe |

Sizes in complex128 at the reference point, from the shapes in `_sweep.py`:

| Object | Size | Scales as |
|---|---|---|
| One core of `M` | 4 GB | `D_M^2` |
| All of `M` | 205 GB | `d D_M^2` |
| One environment `C_j`, one projected environment `S` | 8 GB | `l D_M` |
| All environments | 410 GB | `d l D_M` |
| Largest contraction intermediate | 130 GB | 16 environments |
| Sketch, `Q`, `eta` at one site | 1 GB each | `p l^2` |
| Output train | 51 GB | `d p l^2` |

The work is about `1e16` multiply-adds per compression, two thirds of it in the
right-to-left sketch (`rtl_m`): about 1.5 h on one A100 in complex128. The current
sweep cannot run this problem: it copies every core to the device up front, keeps
every environment on the device, and evaluates each contraction in one piece.

## Requirements

1. Inputs are any `Sequence` of per-site array-likes with `.shape`, `.dtype` and
   `np.asarray` support: NumPy arrays, `np.memmap`, zarr or HDF5 datasets. Each
   core is read when the sweep needs it, never all at once. In a bra stack the MPOs
   are transposed lazily (`SwappedLegs`), so sites without `swapaxes`, such as zarr
   and HDF5 datasets, work there too.
2. The reference problem runs on one A100-40GB with 300 GB of host memory and
   node-local NVMe.
3. The GPU and host memory budgets are detected automatically, with an explicit
   override.
4. Environments that fit nowhere else spill to a scratch directory on local disk.
5. Small problems keep today's behaviour and performance: one batch, everything on
   the device.
6. The CPU and GPU paths stay in sync through `src_method.utils._backend`.

## Non-goals

- Distribution over several GPUs or nodes (Phases 2 and 3).
- Any change to the mathematics, to the random draws or to `truncated_qr`.
- Oversampling of the sketch.
- Writing the output train to disk: at 51 GB it fits the host budget.
- Caching the small layers `N`, `V`, `U` on the device between passes.
- Recomputing environments from checkpoints: with local NVMe, spilling is cheaper.

## Architecture

Five private modules under `src/src_method/`:

| Module | Role |
|---|---|
| `_sweep.py` | Driver: `sweep()` runs the left-to-right and right-to-left passes over the parts below. |
| `_kernels.py` | Site kernels: the einsum equations (moved from `_sweep.py`), the compiled-contraction cache, and three batched per-site functions. |
| `_plan.py` | Planner: `Resources`, budget detection and `make_plan()`. |
| `_store.py` | Environment store: environments on the GPU, in host memory or on disk. |
| `_sites.py` | Site source: lazy, per-site padded access to the cores of each layer. |

The site kernels restrict the current contractions to a slice:

| Kernel | Current equation | Batched over |
|---|---|---|
| `env_step(C_b, omega_b, cores)` | `ltr` | sketch columns `b` of `C_{j+1}` |
| `sketch_step(C_b, cores, S)` | `rtl_m` | sketch columns `b` of the sketch |
| `project_step(eta_b, cores, S)` | `rtl_s`, and `first` at site 0 | rows `b` of the new `S` |

Data flow of `sweep()`:

- Left-to-right: for each site `j` and each batch `b`, `env_step` reads `C_j[b]`
  from the store and writes `C_{j+1}[b]` to it, while the site source prefetches
  site `j + 1`.
- Right-to-left: for each site `j`, `sketch_step` assembles the sketch from batches
  of `C_j`, `truncated_qr` runs unchanged on the whole sketch, `project_step`
  builds the new `S` in row batches, and the store drops `C_j`. The site source
  prefetches site `j - 1`.
- Each `omega_j` is drawn in full on the host, as today, and sliced per batch, so a
  seed gives the same result as today up to floating-point rounding.

## Public API

A new keyword-only argument on `src`, passed through by `apply` and `compress`:

```python
def src(*trains, chi_out, cutoff=0.0, dtype=np.float64, seed=None,
        device="cpu", resources: Resources | None = None) -> list[NDArray]: ...
```

`Resources` is a frozen dataclass defined in `_plan.py` and exported from
`src_method`:

```python
@dataclass(frozen=True)
class Resources:
    gpu_memory: int | str | None = None
    host_memory: int | str | None = None
    scratch_dir: str | os.PathLike[str] | None = None
```

- `gpu_memory`, `host_memory`: `None` detects the budget; otherwise a byte count or
  a size string. `"36GB"` is `36e9` bytes, `"36GiB"` is `36 * 2**30`. Invalid
  values raise `ValueError` at construction, values of another type `TypeError`.
- `scratch_dir`: `None` uses `tempfile.gettempdir()`, which honours `$TMPDIR`; the
  chosen path is logged whenever the disk tier is used.
- On the CPU backend `gpu_memory` is ignored and `host_memory` covers both the
  working set and the in-memory tier.

A configuration object rather than three keywords leaves room for the communicator
settings of Phase 2.

## Planner

`make_plan(site_shapes, chi_out, dtype, budgets) -> Plan` is a pure function of the
padded core shapes, read from `.shape` without loading data, of `chi_out`, of the
dtype and of the resolved budgets. `Plan` is a frozen dataclass holding, per site,
the batch size and contraction path of every kernel and the environment tier, plus
the prefetch depth.

### Memory model

With `e` bytes per element, `A_j` and `B_j` the products of the left and right bonds
of site `j` over all layers, and `p` the size of the output physical legs:

- Fixed per site: the cores of site `j` (twice when the next site is prefetched),
  `S` (`l B_j e`) and the new `S` (`l A_j e`), and the sketch, `Q` and `eta`
  (`p l^2 e` each).
- Per batch column of a kernel: its input and output slices and the peak of live
  intermediates, found by walking the `opt_einsum` path and tracking which tensors
  are alive at each pairwise step.
- Staging: two batches in and two batches out, so that transfers overlap compute,
  unless the environments of the site stay on the GPU.

Costs are in bytes of the working dtype, `np.result_type(dtype, *core dtypes)`:
the sketches promoted by the cores, as the contractions promote them.

The prefetch depth is one site. If the fixed memory with a prefetched site leaves
no room for a batch of one column, the planner sets the depth to zero before
giving up.

### Batch sizes

For each kernel, a binary search over `[1, l]` finds the largest `b` whose cost,
fixed memory, staging and the peak walked from the `opt_einsum` path for that `b`,
fits the budget; `b` is rounded down to a multiple of 32 when `b >= 32` and the
rounded batch still fits. Every batch returned has had its cost checked, and the
path walked is the one the kernels run. No memory limit is passed to `opt_einsum`:
a path that cannot meet it degrades into one large einsum. Batches over the rows of
`eta` are planned for `l` rows, an upper bound on the rank after truncation.

If `b < 1` for any kernel, planning raises `MemoryError` naming the site, the kernel
and the missing bytes: the site working set exceeds the GPU budget, which is the
case Phase 3 addresses.

### Environment tiers

The right-to-left pass reads the environments in reverse order of writing, so the
newest sites go to the fastest tier:

1. Plan the batches at a preferred size of `min(l, 512)` columns.
2. The GPU budget left over holds the environments of the highest `j`.
3. Host memory holds the next ones, within the host budget minus the output train
   (`p l^2 e` per site) and the staging buffers.
4. The scratch directory holds the rest, within its free space.

If the environments do not fit on disk either, planning raises `MemoryError`.

### Budget detection

Explicit `Resources` fields are used as given.

- GPU: `cupy.cuda.runtime.memGetInfo()` free memory plus the free bytes of CuPy's
  memory pool, minus `max(10%, 1 GiB)` for cuBLAS/cuSOLVER workspaces and
  fragmentation. During the sweep the pool is capped at the budget with
  `set_limit`, so an overshoot fails at once; the previous limit is restored
  afterwards.
- Host: `MemAvailable` from `/proc/meminfo`, falling back to `os.sysconf`, minus
  10%. It is measured at call time, so an `M` the caller holds in memory is already
  excluded.
- Disk: `shutil.disk_usage(scratch_dir).free`, minus 5%.

The plan is logged at `info` (batch sizes, tiers per site, estimated peaks), and at
the end of the sweep the stall time, the seconds spent waiting for input cores and
for environments. The pool high-water mark is logged at `debug`.

### Reference plan

At the reference point on an A100-40GB: about 36 GB of budget, about 28 GB fixed
(both `S` at 8 GB each, the core of `M` and its prefetched successor at 8 GB, the
sketch, `Q` and `eta` at about 3 GB), about 8 GB for batches, about 130 MB per
column in `sketch_step`, hence batches of about 64 columns. The environments of a
few sites stay on the GPU, most of the rest in host memory, the remainder on NVMe.

## Site source

`pad()` in `_tensor_train.py` splits into `padded_shape(shape, kind, i, last)` and
`pad_site(array, kind, i, last)`; `pad()` becomes a loop over `pad_site`, so the
padding rules stay in one place.

`SiteSource(layers, kind, xp, depth)`:

- `source.prefetch(j)` queues site `j` on one background thread, which only does
  host work: `np.asarray` of each core (where memmap, zarr and HDF5 read) and a
  copy into a pinned buffer.
- `source[j]` waits for that thread if needed, then issues the asynchronous copies
  of site `j` to the device on the current (compute) stream, and records an event
  that frees the pinned buffer once the copies are done. On the host backend it
  returns the padded arrays, with memmaps read in full.
- The left-to-right pass prefetches `j + 1`, the right-to-left pass `j - 1`.
- A ring of `depth + 1` pinned buffers sized for the largest site is allocated once.
  The host copies count against the host staging budget.

## Environment store

`EnvironmentStore(plan, xp, scratch_dir)` is a context manager:

```python
store.put(j, lo, hi, x)   # columns lo:hi of C_j, from the device
store.get(j, lo, hi)      # device array; the compute stream waits on its arrival
store.prefetch(j)         # start bringing the first batches of site j to the device
store.drop(j)             # release the memory or delete the file
```

| Tier | Storage | Write | Read |
|---|---|---|---|
| GPU | one device array `(l, A_j)` per site | slice assignment | view |
| Host | one pageable array per site | device to pinned buffer on the copy stream, then a background thread copies into the array | a background thread copies into a pinned buffer, then an asynchronous copy to the device on the compute stream |
| Disk | one raw C-order file per site, `<scratch>/src-<pid>-<uuid>/env-<j>.bin`, sketch index outermost | background thread writes the pinned buffer at offset `lo A_j e` | `preadv` into a pinned buffer on a background thread, then an asynchronous copy to the device on the compute stream |

- Pinning hundreds of GB is too costly, so the host tier uses pageable arrays and a
  pinned staging ring.
- Because the sketch index is outermost, every batch is one contiguous byte range.
- After each disk write, `os.posix_fadvise(..., POSIX_FADV_DONTNEED)` on Linux keeps
  the spill from evicting the page cache.
- One writer thread and one reader thread. A map from `(j, batch)` to futures
  ensures that a batch is never read before its write completes.
- The staging ring has the size the planner reserved; `put` blocks when it is full.
- On the CPU backend the GPU and host tiers coincide, leaving memory and disk;
  streams are no-ops, and the threads still overlap disk I/O with compute because
  NumPy's BLAS releases the GIL.

## Backend helpers

New helpers in `src_method.utils._backend`, so the algorithms stay backend-agnostic:

- `Stream` and `Event`: `cupy.cuda.Stream(non_blocking=True)` and CUDA events on
  CuPy; synchronous no-ops on NumPy.
- `pinned_empty(shape, dtype, xp)`: `cupyx.empty_pinned` on CuPy, `np.empty` on
  NumPy.
- `to_device_async(host, xp, stream)` and `to_host_async(device, out, stream)`:
  asynchronous copies on CuPy, plain copies on NumPy.
- `device_memory(xp)`, `device_pool_bytes(xp)`, `device_pool_limit(xp, budget)` and
  `host_memory_available()` for the budgets, the high-water mark and the pool cap.

All kernels run on one compute stream. Device-to-host copies run on a separate
copy stream, which first waits on an event recorded on the compute stream after the
kernel that wrote the batch; the batch stays referenced until its copy is done.
Host-to-device copies run on the compute stream itself: CuPy's pool reuses a block
freed on a stream for the next allocation on that stream, so a copy issued on
another stream could overwrite memory that queued kernels still read, while on the
compute stream the copy is ordered with them. These copies are small against the
compute; the slow part, reading from disk and staging, overlaps compute on the
background threads. The host blocks only to reuse a staging buffer.

## Error handling and cleanup

- Exceptions in background threads are stored in their futures and re-raised by the
  next `get`, `put` or `source[j]`, so the sweep stops at the next step.
- A full disk raises `OSError` naming the path, the bytes needed and the planner's
  estimate.
- `sweep()` uses `with` blocks; a `finally` removes the scratch subdirectory, frees
  the pinned buffers and restores CuPy's pool limit. A `weakref.finalize` covers a
  normal interpreter exit. A killed process leaves its directory behind; the
  per-process name makes it identifiable, and the user docs say so.
- Data movement never touches values, and the inputs are never mutated.

## Testing

Unit tests, CPU only and fast:

1. Planner: a small problem gives one batch and the GPU tier for every site; tight
   budgets shrink the batches, which stay multiples of 32; tiers are assigned
   newest first; an infeasible problem raises `MemoryError` naming the site and the
   missing bytes; the live-memory walk matches a hand-checked path.
2. Budgets: parsing of `"36GB"`, `"36GiB"`, integers and invalid strings; detection
   with `/proc/meminfo`, `memGetInfo` and `disk_usage` replaced by fakes.
3. Site source: `padded_shape` agrees with `pad` for every kind and position; a
   read-counting `Sequence` wrapper shows that each site is read lazily and once per
   pass; an end-to-end run on `np.memmap` inputs.
4. Environment store: `put` then `get` round-trips on every tier, including batch
   edges and a short last batch; `drop` deletes the file; the scratch directory is
   removed after an exception; an exception injected in a worker is re-raised on the
   next call; a simulated `ENOSPC` gives the documented error.
5. Site kernels: batched equals unbatched to `rtol=1e-12` in float64 and complex128,
   including a batch of one and batch sizes that do not divide `l`.

Integration tests on the CPU: a depth-4 MPO stack with 6 sites and small bonds, run
with the default `Resources` and with a tiny `host_memory` that forces small
batches and the disk tier, agrees to `1e-10` for the same seed, in the relative
Frobenius norm of the dense operators. The cores are not compared: batching changes
the rounding, and with it the cores of an ill-conditioned sketch, but not the
operator they represent. `apply` and `compress` pass `resources` through. The existing tests pass unchanged, with no
stray `ResourceWarning`, since warnings are errors in this suite.

GPU tests in `tests/test_gpu_backend.py`, skipped without CuPy: the same comparison
with budgets derived from a plan made with `make_plan`, chosen to reach the device,
host and disk tiers with small batches, which exercises the streams and the pinned
staging. The run succeeds under the pool cap set to the budget, so its peak stays
within it; the pool limit is restored after the sweep, also after an exception.

## Acceptance on the cluster

A new `benches/large/` benchmark, run by hand. Its generator writes `N`, `V`, `U` and
`M` site by site as `.npy` memmaps on NVMe, so `M` is never whole in memory. The run
uses automatic resources and records the plan, the time per pass, the pool
high-water mark, the host peak, the bytes spilled and the stall time, i.e. the time
the host waits on transfers or disk, which the store and the site source log.

Pass criteria:

1. The reference problem completes on one A100-40GB within 300 GB of host memory
   and local NVMe.
2. The pool peak stays within the GPU budget and the host peak within the host
   budget.
3. The stall time is below 10% of the wall time.
4. On a medium problem (`D_M = 1000`, `l = 500`), runs with GPU budgets of 40 GB and
   80 GB agree to `1e-10` in relative Frobenius norm.

## Documentation

- A user page `docs/large-problems.md`, added to the `nav` of `mkdocs.yml`:
  `Resources`, budget detection, the scratch directory and its cleanup caveat, and
  how to read the logged plan.
- `README.md` mentions `Resources`.
- `docs/developer-guide/testing.md` explains how to force the batched and disk-tier
  paths with small budgets.

## Forward compatibility with Phases 2 and 3

- Phase 2 splits the sketch index over the GPUs of a node. The driver hands
  `env_step` and `sketch_step` the columns owned by the GPU instead of all columns,
  and inserts two all-gathers per site in the right-to-left pass: the sketch
  columns after `sketch_step`, and the rows of `S` after `project_step`. The
  kernels, the store, the site source and the planner are reused; the planner
  receives the number of owned columns.
- Phase 3 splits the bond of `M` across nodes. The site source reads the node's
  slice of each core of `M`, the kernels compute partial results over that slice,
  and the driver adds the reductions of the distributed algorithm. The composite
  bond must then keep the `M` index outermost so that the slices are contiguous.

## Risks

- The memory model may underestimate the peak: cuTENSOR and cuBLAS workspaces and
  pool fragmentation are covered only by the margin. The pool limit turns an
  underestimate into an immediate error, and the GPU tests check the model.
- The planner assumes the peak grows with the batch; where a different path at a
  larger batch has a smaller peak, the binary search may settle on a smaller batch
  than possible. Any batch it returns fits.
- `ndarray.get(..., blocking=False)` needs CuPy 13, the floor of the `gpu-nvidia`
  extra.
- zarr and HDF5 decompression may hold the GIL and slow the prefetch thread;
  `np.memmap` of raw `.npy` files does not.
- `set_limit` changes process-wide state of CuPy's default pool; concurrent CuPy
  work in the same process sees the cap during the sweep.
