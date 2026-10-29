# SRC on 8x RTX PRO 6000 (PCIe, complex64): scaling and out-of-core

These runs replace the Leonardo results (4x A100, NVLink, complex128, Lustre) of the
paper's multi-GPU scaling and out-of-core experiments. The reference size is 50
sites, depth 4, `D_M = 4000`, `chi_out = 2000`. The second size is `D_M = 1000`,
`chi_out = 500`. Everything runs in complex64 unless stated. One timed job ran on
the node at a time (`job.sh` takes a `flock`), each with `CUPY_TF32=0` and
`NVIDIA_TF32_OVERRIDE=0`. `analyze.py` regenerates every table below from `raw/`
and `logs/`.

## Provenance

| | |
|---|---|
| Node | Nebius VM, 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GiB each), PCIe 5.0 x16, no NVLink |
| Topology | GPUs 0-3 on socket/NUMA 0, 4-7 on socket 1; pairs `NODE` within a socket, `SYS` (across UPI) between; no shared PCIe switch |
| CPU / RAM | 2x Xeon 6776P (48 cores each), 1.84 TB |
| Storage | no NVMe exposed; 300 GB virtio root disk, `dd` direct I/O 64 GiB: write 0.153 GB/s, read 0.152 GB/s; `/dev/shm` tmpfs 859 GB |
| Software | src-method 0.5.2.dev2+gdc6ecf6df, CuPy 14.2.0, NumPy 2.4.4, opt_einsum 3.4.0, Python 3.14.7 |
| CUDA | runtime 13.2 (13020), driver 580.173.02 (driver API 13000), nvcc 13.0 |
| Commits | step 2 at `a49b19d`; steps 3-6 at `e9ab83c` (in each log's header) |
| Dates | 2026-10-28 19:50 to 2026-10-29 06:00 UTC |

The probes are in `env.json`, with raw text in `node/` and `raw/probe.json`. The
TF32 control matches the strict setting: a complex64 4096 matmul has error 8.1e-7
with TF32 off and 2.9e-4 with TF32 forced on. Measured GEMM rates are 84 TFLOP/s in
complex64 and 1.53 in complex128.

## Deviations from the plan

- **No NVMe.** The disk tier is the virtio root disk at about 0.15 GB/s, which is
  one to two orders of magnitude slower than a local NVMe. The stacks live in
  `/dev/shm` and are memory-mapped, so the `ru_maxrss` host peaks below include up
  to 92 GB of stack pages.
- **The stack generator changed** (`a49b19d`). It drew complex entries with twice
  the variance of real ones. That makes the environments grow about 8x per site,
  and every complex64 output overflowed to NaN near site 43. Complex entries now
  have unit variance. Plans, flops and complex128 timings are unchanged, but these
  stacks are not the ones used on Leonardo.
- **Plan (b) uses 16 GB, not 8 GB.** At the reference size the sketch alone needs
  10.8 GB at a batch of one row, so the planner rejects 8 GB. 16 GB is about the
  smallest budget that plans.
- **Plan (c) ran twice.** (c1) ran as is. (c2) ran under a `MemoryMax` cap to
  squeeze the page cache.
- **Step 6 is supplementary and was not requested.** It reruns G = 4 and G = 8 on
  per-thread default streams, after step 4 traced the communication loss to the
  legacy null stream.

## Step 1: single-precision stacks

`bench_large.py generate --dtype` accepts float32, float64, complex64 and complex128.
The complex64 stack is the complex128 stack rounded, from the same draws
(`tests/test_bench_large.py`). Run records now carry the dtype, `eps` and the QR
distance in units of it.

## Step 2: precision (D_M = 1000, chi = 500, 50 sites, 1 GPU)

| Comparison | QR distance |
|---|---|
| complex64 vs complex128, same stack and seed | 6.40e-5 (537 eps32) |
| complex128 seed 0 vs seed 1 | 1.43 |
| complex128 repeat; complex64 repeat | bit-identical; bit-identical |

| | Wall (s), runs | LTR / RTL (s) |
|---|---|---|
| complex128 | 543.2, 540.1, 541.6 (seed 1) | about 128 / 410 |
| complex64 | 20.3, 20.2 | 4.8 / 13.8 |

Complex64 is 27x faster. Its distance to complex128 is 6.4e-5. That is above the
1e-5 rule of thumb, but 2.2e4 times smaller than the sketch's own seed-to-seed
spread. The spread is O(1) because this random stack is far from compressible at
`chi = 500`.

## Step 3: 1-GPU reference and the prediction

Measured, traced, at the reference size: 843.4 s in all, with LTR 236.3 s and RTL
603.7 s. The RTL kernels run at 63.5 TFLOP/s, 75% of the GEMM rate. The plan puts
7 environments on the device, 43 in host memory and none on disk.

`prediction.md` was committed (`e9ab83c`) before any multi-GPU run. It predicted
efficiencies of 98%, 92% and 80% at G = 2, 4 and 8.

## Step 4: strong scaling

### D_M = 4000, chi = 2000

| G | Wall r1 / r2 (s) | Speed-up | Efficiency | Predicted | LTR / RTL (s) | Stalls sites / envs (s) | Tiers d/h/disk | Sketch batch | RTL TFLOP/s per GPU | QR distance r1 / r2 (eps) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 846.4 / 847.6 | 1.00 | 100% |  | 238.7 / 604.2 | 5.3 / 281.7 | 7/43/0 | 544 | 60.9 | 0 / 0 |
| 2 | 454.1 / 454.7 | 1.86 | **93%** | 98% | 128.9 / 324.5 | 28.6 / 96.8 | 15/35/0 | 992 | 56.7 | 1.99e-04 (1673) / 1.99e-04 (1671) |
| 4 | 259.2 / 258.1 | 3.27 | **82%** | 92% | 70.2 / 187.5 | 38.3 / 30.7 | 33/17/0 | 480 | 49.1 | 1.99e-04 (1673) / 1.99e-04 (1669) |
| 8 | 169.5 / 169.1 | 5.00 | **63%** | 80% | 38.0 / 127.9 | 34.9 / 0.0 | 50/0/0 | 224 | 36.2 | 2.01e-04 (1684) / 2.01e-04 (1684) |

Efficiency uses the median of the two reps. LTR and RTL are the slowest device's
pass times. RTL TFLOP/s per GPU divides each rank's traced RTL flops by the untraced
RTL pass time. The QR distance is to the 1-GPU output of the same rep.

### D_M = 1000, chi = 500

| G | Wall r1 / r2 (s) | Speed-up | Efficiency | LTR / RTL (s) | Tiers d/h/disk | Sketch batch | QR distance (eps) |
|---|---|---|---|---|---|---|---|
| 1 | 20.3 / 20.3 | 1.00 | 100% | 4.8 / 13.8 | 50/0/0 | 480 | 0 |
| 2 | 10.6 / 10.4 | 1.93 | 96% | 2.5 / 7.9 | 50/0/0 | 224 | 8.45e-05 (709) |
| 4 | 7.7 / 7.6 | 2.64 | 66% | 1.7 / 5.7 | 50/0/0 | 96 | 7.79e-05 (654) |
| 8 | 9.7 / 9.7 | 2.10 | 26% | 2.8 / 6.2 | 50/0/0 | 32 | 7.84e-05 (658) |

At this size 8 GPUs are slower than 4. Each GPU's share of an RTL site at G = 8 is
about 35 ms of compute. That is the same order as one allgather of `S` (256 MB)
at the in-sweep rate below, plus the host barriers of eight threads. This size
was not traced.

### Accuracy across G

The multi-GPU outputs sit 2.0e-4 from the 1-GPU output at the reference size
(about 1700 eps32), and 8e-5 at D_M = 1000. This is not specific to running on
several GPUs. On one GPU, plan (b) changes only the sketch batch (64 rows instead
of 544), and its output sits the same 2.0e-4 from plan (a). Any change in
summation order moves this output that far, because its truncated spectrum has no
gap at `chi`. At G = 8 the two reps' distances to the 1-GPU output match to every printed
digit. At G = 2 and 4 they match to 3 digits but not bitwise. The difference is
that rep 1 hit an out-of-memory split of its sketch and projection batches, which
the library recovers from, and rep 2 did not.

### Prediction against measurement, and where the time goes

From the traces at G = 1, 2, 4 and 8, rank 0's right-to-left pass in GPU seconds
breaks down as follows:

| G | Span | Kernels | TFLOP/s in kernels | allgather S | allgather cores | gather + scatter | QR | Other | Kernels x G |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 596.2 | 579.7 | 63.5 | 0.0 | 0.0 | 0.0 | 2.8 | 13.7 | 579.7 |
| 2 | 316.6 | 291.3 | 63.1 | 5.0 | 2.5 | 0.7 | 2.8 | 14.5 | 582.5 |
| 4 | 182.9 | 145.7 | 63.2 | 13.6 | 6.4 | 1.3 | 2.7 | 13.1 | 582.9 |
| 8 | 121.3 | 70.7 | 65.5 | 24.8 | 12.8 | 1.6 | 2.7 | 8.7 | 565.9 |

The left-to-right pass on rank 0 has a span of 243.8 / 132.8 / 73.9 / 40.0 s, kernels
of 220.5 / 109.9 / 55.2 / 26.9 s and core allgathers of 0 / 2.5 / 6.9 / 12.9 s.

**The computation scales perfectly.** Kernel time times G stays at 566-583 s, and
the kernels keep 63-65 TFLOP/s down to the 224-row sketch batch at G = 8. All of the
efficiency loss is time that does not divide by G:

1. **The collectives run at about half the probed bandwidth.** The fastest rank's
   per-site allgather of `S` is the low bound, because every rank waits for the
   slowest in it:

   | G | allgather S, fastest rank (ms/site) | In-sweep GB/s per receiver | `DeviceGroup.allgather` alone, null / per-thread stream | Probe |
   |---|---|---|---|---|
   | 2 | 105 | 19.4 | 19.5 / 34.8 | 37.2 |
   | 4 | 279 | 11.0 | 11.0 / 15.3 | 21.3 |
   | 8 | 525 | 6.8 | 6.6 / 11.3 | 15.7 |

   `allgather_check.py` runs `DeviceGroup.allgather` alone at the same payloads. It
   reproduces the in-sweep rate exactly, so the cause is not contention with the
   sweep. The sweep issues its kernels and peer copies on CuPy's current stream,
   which by default is the legacy null stream. On the null stream a peer copy
   into GPU 0 serializes with the copy into GPU 1. Measured with 1 GB and 2 GB
   blocks: 20.7 GB/s bidirectional on null streams, 37.3 GB/s on non-blocking
   streams, 41.3 GB/s one-way either way. The probe used non-blocking streams,
   which is why the model was optimistic. With per-thread default streams
   (`CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`) the collective reaches 34.8, 15.3 and
   11.3 GB/s. What remains below the probe at G = 4 and 8 comes from the
   collective's events and barriers; it was not isolated further.
2. **The collectives grow with G, the work does not.** Each receiver pulls
   `(G-1)/G` of `S` (4.1 GB per site) and of each core (2.05 GB). From 4 to 8 GPUs
   the payload grows by 17% while the per-receiver bandwidth drops from 21.3 to
   15.7 GB/s in the probe. **The topology contributes here.** G = 4 uses GPUs 0-3,
   all on one socket (`NODE`). G = 8 adds GPUs 4-7, so half of every GPU's peers
   are across UPI (`SYS`, 34-39 GB/s per pair against 41 GB/s within a socket),
   and all eight share the root complexes. From 4 to 8, rank 0's allgather time
   grows from 20.0 to 37.6 s in RTL and from 6.9 to 12.9 s in LTR. Together the
   collectives take 52 s of the 165 s traced run at G = 8, against 24 s in the
   model.
3. **The left-to-right "other" does not divide by G.** At G = 1 and 2 the GPU idles
   for 23.3 and 20.4 s of LTR. That is consistent with waiting on environment
   copies to host memory, the only other transfer in that pass. The
   model divided this by G as if it were compute. It accounts for most of the 2-GPU
   shortfall (454 s measured against 429 s predicted). From G = 4 the environments
   move onto the devices (33 and then 50 of them), so the term fades.

The model underestimated the 4-to-8 loss mainly through point 1. Its formula was
right, but the bandwidths came from the wrong stream type.

## Step 5: out-of-core tiers (1 GPU, reference size)

| Plan | Tiers d/h/disk | Planned device / pool (GB) | Planned host / max RSS (GB) | Spilled (GB) | Disk written / read (GB) | Wall (s) | LTR / RTL (s) | Stalls sites / envs (s) | Bit-identical to (a) |
|---|---|---|---|---|---|---|---|---|---|
| (a) default | 7/43/0 | 91.1 / 81.7 | 223.6 / 298.1 | 0.0 | 0.0 / 0.0 | 842.3 | 238.6 / 601.2 | 5.0 / 280.1 | -- |
| (b) `--gpu-memory 16GB` | 1/49/0 | 15.9 / 22.8 | 246.1 / 318.8 | 0.0 | 0.0 / 0.0 | 799.3 | 207.4 / 589.2 | 0.0 / 572.6 | no (2.0e-04) |
| (c1) `--host-memory 50GB` | 7/0/43 | 91.1 / 81.7 | 47.5 / 144.0 | 176.1 | 176.1 / 171.7 | 2859.1 | 2255.1 / 600.8 | 4.9 / 2326.0 | yes |
| (c2) as (c1), `MemoryMax=142G` | 7/0/43 | 91.1 / 81.7 | 47.5 / 144.0 | 176.1 | 176.1 / 220.1 | 3058.2 | 2265.7 / 789.1 | 5.3 / 2616.8 | yes |

"Pool" is CuPy's pool size at the end of the run. Max RSS includes the stack's
mapped pages (up to 92 GB). Disk traffic is the process's block-layer I/O, from
`time -v`.

- **(b) completes in 799 s against 842 s, but overruns its budget.** It is 5%
  faster than the default plan with 1/6 of the device memory. However, CuPy's pool
  reached 22.8 GB against a 15.9 GB planned peak under the 16 GB budget, 43% above
  the plan. The pool size includes cached free blocks, so it is an upper bound on
  the live peak. The planner should be checked against that before the paper claims
  budgets are respected at small sizes. Its output is not bit-identical to (a)
  because the sketch batch differs (64 rows instead of 544). The distance, 2.0e-4,
  is the same as across G.
- **(c) spills to disk.** All 43 host environments spill, 176.1 GB in total. The
  run wrote exactly that and read 171.7 GB back from the block device. The page
  cache did not absorb the spill, despite the 1.8 TB of RAM: the store's
  `POSIX_FADV_DONTNEED` drops the pages once written back. The planned host peak
  of 47.5 GB is consistent with a measured RSS of 144 GB minus the up to 92 GB of
  mapped stack.
- **The disk cost is entirely in the writes.** Relative to (a), LTR grows by
  2017 s for 176.1 GB, an effective 87 MB/s, or 57% of the direct-I/O `dd` write
  rate. At the full `dd` rate the writes alone would take 1151 s. RTL in (c1)
  matches (a) (600.8 s against 601.2 s): the 171.7 GB of reads, at least
  0.29 GB/s, hide entirely behind the computation.
- **(c2) shows the effect of a cap.** The cap was set from (c1)'s RSS, but the
  tmpfs stack pages are charged to the cgroup that created them. So the cap
  squeezed rather than removed the page cache. Readahead pages were evicted and
  re-read (220.1 GB read for 176.1 GB stored), adding 188 s to RTL. The cgroup's
  OOM killer did not fire.
- **(a), (c1) and (c2) are bit-identical.** The tier changes where environments
  live, not what is computed.

## Step 6 (supplementary): per-thread default streams

The same code was rerun with `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`, one fresh
process per run:

| Size | G | Step 4, null stream (s) | Per-thread stream (s) | Efficiency | LTR / RTL (s) | QR distance to (a) |
|---|---|---|---|---|---|---|
| m4000 | 8 | 169.3 | 145.6 | **73%** | 32.9 / 108.3 | 2.01e-04 |
| m4000 | 4 | 258.7 | 246.3 | **86%** | 67.9 / 175.5 | 1.99e-04 |
| m1000 | 8 | 9.7 | 12.1 | 21% | 3.0 / 7.1 | |
| m1000 | 4 | 7.7 | 8.4 | 60% | 1.6 / 5.1 | |

The outputs are as far from (a) as in step 4. At the reference size the G = 8 run
gains 14%, from 63% to 73% efficiency, against
the 80% predicted. The m1000 rows are not comparable to step 4. In step 4,
`scaling` reuses one process, so G = 4 and 8 run after warm-up. Here each run
starts cold and pays CUDA context and library set-up on every GPU, which is
seconds at a 10 s scale. Making the sweep's streams non-blocking (or per-thread)
in `src_method` is the obvious follow-up. It needs a separate change with its own
tests; this branch only measures it.

## Files

- Scripts, in run order: `job.sh`, `probe_env.py`, `make_env.py`,
  `step2_precision.sh`, `step3_reference.sh`, `predict.py`, `step4_scaling.sh`,
  `trace.py`, `step5_tiers.sh`, `distances.py`, `allgather_check.py`,
  `step6_ptds.sh`, `analyze.py`.
- Run records, one JSON line per run: `raw/step2_precision.jsonl`,
  `raw/step4_m{4000,1000}.jsonl`, `raw/step5_tiers.jsonl`,
  `raw/step6_ptds_m{4000,1000}.jsonl`.
- Distances: `raw/step{2,5,6}_distances.jsonl`.
- Traces: `raw/trace_g{1,2,4,8}_m4000.json`.
- Probes: `raw/probe.json`, `raw/allgather_check{,_ptds}.jsonl`, `env.json` and
  `node/`.
- Logs: `logs/`, one per timed job, each with its command, git commit and
  `time -v` output.
