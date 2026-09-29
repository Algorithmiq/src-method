# Design: SRC sweep over the GPUs of a node (Phase 2)

- Status: approved; refined while planning (see the plan's "Refinements of the spec")
- Builds on: Phase 1, the out-of-core sweep
  ([design](2026-09-25-src-out-of-core-design.md)).
- Follow-up: Phase 3, the bond split across nodes; see
  [Forward compatibility](#forward-compatibility-with-phase-3).

## Goal

Run one `src` sweep on the GPUs of a node, splitting the sketch index among them, so
that the reference problem of Phase 1 finishes close to `G` times faster on `G`
GPUs. The mathematics, the random draws and the result do not change.

## Background

The Khatri-Rao sketch makes the `chi_out` (`l` below) sketch columns independent in
the environment pass and in the sketch step, so they can be split among GPUs:

- GPU `g` owns a subset `J_g` of the sketch columns and builds only those columns
  of every environment, with no communication.
- At every site of the right-to-left pass, the sketch columns are all-gathered, the
  thin QR is repeated on every GPU, each GPU projects a block of rows of the new
  projected environment `S`, and those rows are all-gathered.
- The cores of `M` are never split: every GPU streams the same cores.

This is Algorithm 5 of the working notes (in `sandbox/`, not versioned), mirrored
to the direction of `_sweep.py`, which builds environments left-to-right and
compresses right-to-left.

Because every GPU holds a full core of `M`, a full `S` and the full sketch, the
per-site working set does not shrink with `G`: the largest `D_M` stays near `10^4`
in complex128 at `l = 2000` on 80 GB GPUs, and only Phase 3 raises it. Phase 2 is
about time to solution. The environments do shrink, to `1/G` per GPU, so at the
reference size they fit in the combined device memory of a node and stop spilling.

## Decisions

| Question | Decision |
|---|---|
| Execution model | one process per GPU (SPMD), launched with `srun` or `mpirun` |
| Collectives | NCCL through `cupy.cuda.nccl`; `mpi4py` for bootstrap and small host messages |
| Public interface | an `mpi4py` communicator in `Resources(comm=...)`; nothing is collective without it |
| Output | rank 0 in memory by default; with `Resources(output_dir=...)`, rank 0 writes `.npy` files and every rank returns memmaps |
| Large inputs | must be file-backed (`np.memmap`, zarr, HDF5), so the page cache shares them among the ranks of a node |
| Success criterion | strong scaling at the reference size |

## Requirements

1. With `Resources(comm=...)`, every rank calls `src` (or `apply`, `compress`) with
   the same arguments and the sweep runs on all of them.
2. The result equals the single-process result for the same seed, up to
   floating-point rounding.
3. Without a communicator, or with a communicator of size one, the sweep is exactly
   the Phase 1 sweep.
4. `mpi4py` and NCCL stay optional: a single-process run imports neither.
5. The distributed logic runs on the CPU backend too, with `mpi4py` on host arrays,
   so that CI can test it without GPUs.
6. A failure must not leave ranks waiting in a collective.

## Non-goals

- The bond split and multiple nodes (Phase 3).
- Sharing in-memory inputs among ranks through MPI shared memory, or one reader per
  node broadcasting cores; large inputs are file-backed instead.
- GPUDirect Storage and compressed inputs; see
  [Future extensions](#future-extensions).
- Heterogeneous GPUs within a node.
- Raising the largest `D_M` or `l` a site can have.

## Public interface

`Resources` gains two optional fields:

```python
Resources(gpu_memory=None, host_memory=None, scratch_dir=None,
          comm=None,          # an mpi4py communicator; None = single process
          output_dir=None)    # write the output cores here as .npy files
```

- `comm` is typed as `Any`, so `mpi4py` is only imported when a communicator is
  given.
- `output_dir` also works without `comm`: the output is written and returned as
  memmaps. It lives in `Resources` so that the signatures of `src`, `apply` and
  `compress` do not change.
- `src`, `apply` and `compress` return `list[NDArray] | None`: rank 0 returns the
  cores and the other ranks `None`; with `output_dir`, every rank returns read-only
  memmaps of the same files, after a barrier.

Contract of a distributed call:

- Before any work, the ranks all-gather a fingerprint of the call: layer shapes and
  dtypes, `chi_out`, `cutoff`, `dtype` and `seed`. On a mismatch, every rank raises
  the same `ValueError`.
- With `seed=None`, rank 0 draws a seed and broadcasts it before the generator is
  created.
- A rank that sees one device (the usual case under `srun`, which sets
  `CUDA_VISIBLE_DEVICES`) uses it; a rank that sees several uses device
  `local_rank mod count`.
- Two-site stacks take the exact path on every rank; the output is handled as above.
- With more than one rank per node, an in-memory input layer (not a memmap) above
  1 GiB is logged as a warning: every rank holds a copy.

## Communicator

A new private module `_comm.py` defines the collectives the driver needs, and three
implementations of them:

```python
class Communicator(Protocol):
    rank: int; size: int              # in the communicator
    local_rank: int; local_size: int  # within the node
    def allgather(self, local: NDArray, axis: int) -> NDArray: ...
    def allgather_objects(self, value: object) -> list[object]: ...
    def bcast_int(self, value: int, root: int = 0) -> int: ...
    def barrier(self) -> None: ...
    def abort(self, code: int) -> None: ...

def make_communicator(comm: Any | None, xp: ModuleType) -> Communicator: ...
```

- `allgather` concatenates the blocks of every rank along `axis`, in rank order;
  blocks may differ in size along `axis`.
- `SingleComm`: identity semantics, used when `comm` is `None` or has size one.
- `MpiHostComm`: `mpi4py` `Allgatherv` on NumPy arrays, for the CPU backend.
- `NcclComm`: `cupy.cuda.nccl` on CuPy arrays, on the compute stream so that the
  collectives are ordered with the kernels. Data travels as bytes, because NCCL has
  no complex type, and blocks are padded to a common size, because NCCL has no
  all-gather with variable counts; the padding and unpadding are pure functions.
  The NCCL unique id is broadcast over `mpi4py`, and the node layout comes from
  `MPI.COMM_TYPE_SHARED`.

The driver makes the same calls in all three cases, which keeps the backend
difference behind one seam, as `AGENTS.md` requires. Phase 3 adds rail and node
communicators as more instances of the same interface.

## Driver

`src` (in `stack.py`) prepares the call, so that the exact path benefits as well:

1. Build the communicator with `make_communicator(resources.comm, xp)` and select
   the device.
2. All-gather the fingerprint; raise the same `ValueError` on every rank on a
   mismatch.
3. Agree on the seed, then create the generator.
4. Run the exact path or the sweep, passing the communicator to `sweep`.
5. Handle the output: rank 0 in memory, or `output_dir` followed by a barrier and
   memmaps on every rank.

In `sweep`:

- Rank `g` owns the sketch columns `J_g = range(g, l, G)`, kept locally as columns
  `0 .. n_g - 1`. Each `omega_j` is drawn in full on every rank, as in Phase 1, and
  sliced as `omega[g::G]`. The environments have shape `(n_g, *left_bonds)`.
- The left-to-right pass is Phase 1's, on the local columns, with no communication.
- In the right-to-left pass, at each site:
  - the local sketch `(eta, up, down, n_g)` is assembled from batches, then
    all-gathered along the column axis into `(eta, up, down, l)`; the columns come
    out permuted, identically on every rank, which leaves the range and hence `Q_k`
    unchanged;
  - the thin QR runs on every rank; with `cutoff > 0`, the rank is decided on rank 0
    and broadcast: `truncated_qr` gains an `agree_rank` callback that receives the
    local rank and returns the one to apply;
  - rank `g` projects a contiguous block of the rows of the new `S`, in batches, and
    the blocks are all-gathered along the row axis;
  - only rank 0 keeps the output core, unpadded as it is produced, and hands it to an
    output sink: in memory, or written to `output_dir`.
- The first site's output core is computed on rank 0 only.

If the GPU QR is not bitwise reproducible, the blocks of `P_k` (the new `S`) come
from copies of `Q_k` that differ at rounding level; the effect is at rounding level
too.

## Planner

Per rank:

- Environment and sketch batches are planned for `n_g` columns and the local
  environment shapes; projection batches for `ceil(l / G)` rows.
- The fixed memory gains the two gather buffers, each about the size of the full
  sketch or the full new `S`. The full sketch, `Q` and `S` stay on every rank.
- The host budget is a share of the node: `(host - output) / local_size` on every
  rank, where `output` is the output train when it is held in memory; rank 0 also
  holds the output. The scratch disk is shared the same way.
- The GPU budget is unchanged: each rank has its own device.

Plans may differ between ranks, by one column and by the output on rank 0; the
collectives do not depend on batch sizes.

## Errors

A collective blocks until every rank joins it, so a failure on one rank must not
leave the others waiting.

- Up to the start of the sweep, errors are agreed: the fingerprint check raises on
  every rank, and after planning the ranks all-gather either an ok or the text of
  their `MemoryError`, and all raise the same error.
- During the sweep, an exception on one rank is logged with its rank, then
  `MPI.Abort(1)` ends the job. NCCL and MPI collectives cannot be cancelled, and a
  clear abort is better than a job that hangs until its time limit.

## Dependencies

- A new extra `mpi = ["mpi4py>=4"]`, which needs an MPI library: a system Open MPI or
  MPICH, the `mpich` or `openmpi` wheels of the mpi4py project, or `impi-rt`. CI
  installs the system MPICH on its Ubuntu runner, and the Nix dev shell provides
  MPICH.
- The `gpu-nvidia` extra gains `nvidia-nccl-cu12`.

## Expected scaling

Per-site budget at the reference size (50 sites, `D_M = 4000`, `l = 2000`,
complex128):

| Item | Cost per site |
|---|---|
| Compute on one A100 | about 110 s |
| Compute on each of 8 A100 | about 14 s |
| The two all-gathers, about 9 GB over NVLink/NVSwitch | 30-40 ms on A100, less on H100 |
| The repeated QR (32000 x 2000, complex) | 0.1-0.2 s |
| Copies of the core of `M` to each GPU, twice per site, over shared PCIe switches | 0.3-0.6 s |
| Reading `M_k` from NVMe, once per node per pass thanks to the page cache | hidden by the prefetch thread |

The parts that do not scale cost about 3-6% at 8 GPUs, so an efficiency near 90% is
expected. On small problems the repeated QR and the collectives weigh more and the
efficiency drops, as expected.

## Testing

Unit tests, CPU only, in the PR suite:

- `SingleComm` semantics, and `make_communicator(None, xp)`.
- The padding and unpadding of uneven blocks for NCCL.
- The cyclic column split and the contiguous row blocks cover every index exactly
  once, also when `l` is not a multiple of `G`.
- The planner plans for `n_g` columns and `ceil(l / G)` rows, includes the gather
  buffers, and shares host memory and scratch disk; rank 0 holds the output unless
  `output_dir` is set.
- `output_dir` in a single process: the memmaps equal the in-memory result and the
  files exist.

MPI tests on the CPU, in the normal suite, skipped without `mpi4py` or `mpirun`:

- Each test launches `mpirun -n 2` or `-n 3` (an uneven split) on a small script in
  `tests/mpi_scripts/`; each rank writes its result to `tmp_path`, and the test
  compares. Every launch has a timeout, so a hang fails the test instead of
  stalling the suite.
- The distributed result equals the single-process one for the same seed, in the
  dense operator to `1e-10`, with and without `cutoff`; with `seed=None` the ranks
  agree.
- Rank 0 returns the cores and the other ranks `None`; with `output_dir`, every rank
  returns identical memmaps.
- A fingerprint mismatch raises `ValueError` on every rank; a planning failure on
  one rank raises `MemoryError` on every rank.
- Tiny budgets force the disk tier on every rank, and every scratch directory is
  removed.
- An exception injected during the sweep aborts the job: nonzero exit status, and a
  log line naming the rank.
- CI installs an MPI library so that these tests run on every PR.

GPU tests in `tests/test_gpu_backend.py`, skipped without two GPUs and `mpirun`:
`mpirun -n 2` with `NcclComm` gives the operator of a single-GPU run, with an odd
`l` to exercise the padding.

## Acceptance on the cluster

`benches/large/` gains an MPI mode, where every rank runs `run` and reports its plan
and times, and a `scaling` command, launched with `srun` for 1, 2, 4 and 8 GPUs,
that records the wall time, the parallel efficiency `T_1 / (G T_G)` and the distance
to the single-GPU result, computed from `output_dir` memmaps with TT inner products.

Pass criteria:

1. The distributed result matches the single-GPU result to `1e-10` in relative
   Frobenius norm.
2. At the reference size, parallel efficiency at least 80% at 8 GPUs on 8xA100 and
   on 8xH100, against Phase 1 on one GPU of the same node, with the 1/2/4/8 curve
   recorded; at least 90% on 2xA100.
3. Without a communicator, or with one rank, the sweep is exactly Phase 1.

## Documentation

- A "Several GPUs" section in `docs/large-problems.md`: launching with `srun` or
  `mpirun`, `Resources(comm=..., output_dir=...)`, file-backed large inputs, how the
  ranks of a node share host memory and scratch disk, and the abort on errors
  during the sweep.
- `README.md` points to it; `docs/developer-guide/dependencies.md` lists the `mpi`
  extra; `docs/developer-guide/testing.md` explains how to run the MPI tests.

## Forward compatibility with Phase 3

Phase 3 splits the bond of `M` across nodes and keeps the sketch split within a
node. The rail communicator (the GPUs with the same index on every node) and the
node communicator become two `Communicator` instances built from `mpi4py`
sub-communicators; the driver adds the reductions of the bond split along the rails
and keeps the Phase 2 all-gathers within the node.

## Future extensions

- **GPUDirect Storage.** cuFile, or KvikIO from Python, moves data between NVMe and
  device memory without a host bounce buffer. It fits the environment spill files
  first: they are private to each rank, written by the store, and their batch
  offsets are multiples of the row size, hence aligned. The site source and the
  disk tier of the store would read into a ring of device buffers instead of pinned
  host buffers. For inputs it bypasses the page cache that the ranks of a node
  share, so it would need one reader per node broadcasting the cores over NCCL;
  `.npy` files would also want 4 KiB-aligned data. Zarr with uncompressed chunks
  and KvikIO's zarr support, with nvCOMP for decompression on the GPU, is the route
  if compressed inputs are ever wanted.
- **Compression.** Lossless compression is not planned: the mantissas of dense
  cores, and in particular of the isometries SRC produces, compress by about
  1.0-1.2, and reading is not the bottleneck. It becomes worthwhile for structured
  operators (MPOs of finite-state machines or of Pauli strings are mostly exact
  zeros), or if a real core of `M` compresses by more than about 1.3 with Blosc
  (Zstd, bitshuffle). Lossy compression would be a numerical decision, not an I/O
  one.

## Risks

- NCCL or MPI availability and configuration differ between clusters; the NCCL
  wheel reduces the risk on the GPU side, and the CPU backend with `mpi4py` keeps
  the logic testable anywhere.
- `MPI.Abort` ends the whole job, including unrelated work in the same MPI program;
  this is documented.
- With eight ranks streaming the same `M` through the page cache, an `M` larger
  than the free host memory is read from disk once per pass per node rather than
  once per pass; the prefetch still hides it at the reference size.
- Rank 0 carries extra host memory (the output) and a little extra work (the first
  site and the output copies); the plan accounts for the memory, and the work is
  small.
- The copies of the cores of `M` to each GPU run on the compute stream (a Phase 1
  refinement) and cost a few percent at 8 GPUs; they are the first optimisation to
  try if the measured efficiency falls short.
