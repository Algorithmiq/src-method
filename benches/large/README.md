# Out-of-core benchmark

`bench_large.py` compresses the stack `N . V . M . U` of Pauli-transfer-matrix MPOs
(physical legs of 4) on one GPU or several, where `M` has a large bond and is read
from disk.

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

# Split the sweep among the GPUs of the node: 1, 2, 4 and 8 of them, in turn.
uv run python benches/large/bench_large.py scaling /local/stack --chi-out 2000 \
    --devices 1 2 4 8 --scratch-dir /local/scratch --results scaling.jsonl
```

`run` logs the wall time, the size of CuPy's pool on each GPU at the end (its high-water mark,
since the pool keeps its blocks), the host peak (`ru_maxrss`), the planned peaks,
the bytes spilled and the tiers. With `--debug` before the command, `src_method` also
logs its plan, the time of each pass and the stall time (`SRC stalls`).

Acceptance of the out-of-core sweep on one A100-40GB, with at least 300 GB of host
memory and node-local NVMe:

1. `run` completes at 50 sites, `D_M = 4000`, `chi_out = 2000`, complex128.
2. The pool size stays within the GPU budget plus the `max(10%, 1 GiB)` margin
   the pool cap allows for fragmentation, and the host peak within the host
   budget.
3. The stall time is below 10% of the wall time.
4. `compare` at `D_M = 1000`, `chi_out = 500` reports a distance below `1e-10`.

`scaling` logs, for each GPU count, the wall time, the speed-up and the parallel
efficiency `T_1 / (G T_G)` against the first count, and the relative distance to
its output; `--results` appends the same as JSON lines. `--devices` with
`run` splits a single run. Acceptance of the multi-GPU sweep, at the reference
size on 8xH100 or 8xH200 (and 4xA100 on Leonardo):

1. Every distance is below `1e-10`.
2. The parallel efficiency is at least 80% on all the GPUs of the node.

`scaling` reports two distances. `distance` comes from inner products and cancels
down to about `sqrt(eps)`, near `1.5e-8` in complex128, so it cannot show `1e-10`.
`QR distance` sweeps a QR over the MPO `a - b` (the direct sum of the two outputs)
and resolves differences down to rounding; the acceptance applies to it.

## Leonardo

The [`leonardo/`](leonardo/) folder holds the Slurm script and the logs; see the
[Leonardo section](../README.md#leonardo) for the environment. Booster nodes have
no local disk and a 10 GB `/tmp`, so the script points `TMPDIR` at
`$CINECA_SCRATCH`, and the stacks live there too (Lustre, not NVMe):

```bash
D=$CINECA_SCRATCH/src-large
sbatch run.sh generate $D/m4000 --bond-m 4000
sbatch run.sh --debug run $D/m4000 --chi-out 2000 --scratch-dir $D/scratch
sbatch run.sh generate $D/m1000 --bond-m 1000
sbatch run.sh --debug compare $D/m1000 --chi-out 500 --small 4GB --large 60GB \
    --scratch-dir $D/scratch
```

`generate` writes the `D_M = 1000` stack (12 GB) in 51 s.

`compare` at `D_M = 1000`, `chi_out = 500`, complex128, on one A100-64GB:

| GPU budget | Wall time (s) | Pool (GB) | Planned device peak (GB) | Host peak (GB) | Tiers (device/host/disk) |
|---|---|---|---|---|---|
| 4GB  | 129.8 | 5.46  | 3.86  | 40.6 | 1 / 49 / 0 |
| 60GB | 109.9 | 60.81 | 58.84 | 40.6 | 50 / 0 / 0 |

The two plans keep opposite tiers yet give bit-identical outputs (relative distance
`0.000e+00`). Spilling 49 of 50 environments to the host costs 18 % of the wall
time. Both pools stay within their caps (budget plus 10 % of the card, 6.9 GB).

## Several GPUs (Leonardo, 4xA100-64GB)

One Booster node: 4 A100-SXM-64GB, 32 cores, 512 GB of memory, CUDA 13 through the
forward-compatible driver (see [`../README.md`](../README.md#leonardo)). The stacks
live on Lustre scratch. `leonardo/run_multi.sh` runs any command with the 4 GPUs
visible to one process, prints the topology and the peer-access matrix, and with
`WARM_DIR=<stack>` reads the stack once beforehand, so that the first run of a job
does not read it cold. Logs and JSON lines are in [`leonardo/logs/`](leonardo/logs/).

### GPU tests

`pytest tests/test_gpu_backend.py` (job 59776456): all 19 tests pass, among them
`test_two_gpus_match_one[split]`, `test_two_gpus_match_one[full]`,
`test_two_gpus_take_device_inputs`, `test_two_gpus_raise_the_first_error` and
`test_too_many_devices_raise`. The four GPUs are fully connected (`NV4`, four
bonded NVLinks per pair) and peer access is on for every pair:

```text
[[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
```

so each GPU uploads a slice of every core and gathers the rest over NVLink.

### D_M = 1000, chi_out = 500

`scaling --devices 1 2 4` (job 59782618), complex128, every environment on the
device in all three runs:

| GPUs | Time (s) | Speed-up | Efficiency | Speed-up vs 109.9 s | Efficiency vs 109.9 s | QR distance |
|---|---|---|---|---|---|---|
| 1 | 131.4 | 1.00 | 100% | | | |
| 2 | 43.6 | 3.01 | 151% | 2.52 | 126% | 2.8e-13 |
| 4 | 32.7 | 4.02 | 101% | 3.36 | 84% | 4.4e-13 |

The single-GPU time depends on whether it is the first SRC run of the job. The
same single-GPU run as the second or third run of a job (job 59783330, separate
processes) takes 109.7 s with a 60 GB budget and 109.9 s with the detected one, as
in the single-GPU `compare` of #59 (109.9 s); the plan
is identical (device peak 58.8 GB, sketch batch 480, every environment on the
device). The extra 22 s of the first run fall in both passes (left-to-right 23.8 s
against 13.7 s, right-to-left 102.5 s against 95.4 s) with the same site stalls
(13.3 s against 13.5 s); its cause is not isolated. The second pair of columns uses
109.9 s.

| GPUs | Left-to-right (s) | Right-to-left (s) | Site stalls (s) | Sketch batch | Planned device peak (GB) |
|---|---|---|---|---|---|
| 1 (second run of a job) | 13.7 | 95.4 | 13.5 | 480 | 58.8 |
| 2, per GPU | 7.02 | 35.9 / 36.4 | 6.7 / 6.8 | 224 | 39.0 |
| 4, per GPU | 3.93 | 28.0 (GPU 0: 28.5) | 3.6 | 96 | 32.3 |

No GPU stalls on environments: they all stay on the device. The pool sizes are
not listed: `scaling` runs every count in one process and CuPy keeps its blocks,
so the pools report the high-water mark of all the runs so far.

Where the time goes:

- The left-to-right pass needs no communication and scales by 1.95 and 3.49 on 2
  and 4 GPUs. The site stalls (the time a GPU thread waits for the loader to hand
  over the next site's cores) shrink in proportion (13.5, 6.8, 3.6 s) and are close
  to the left-to-right time. The timer stops before the upload, and the loader also
  waits for the GPU to release a staging buffer, so a stall cannot tell a slow read
  from a GPU-bound pass; the second is likely here, but it was not verified.
- The right-to-left pass scales by 2.62 on 2 GPUs, which is superlinear, and by 3.35
  on 4. The superlinear part is not explained by the plan or the tiers, which are
  the same on every count; the sketch batch drops from 480 to 224 and 96 columns
  per GPU, and smaller batches may run the contractions more efficiently, but this
  was not isolated. On 4 GPUs the pass takes 28.5 s against 23.9 s for a quarter
  of the single-GPU pass; GPU 0 ends 0.46 s after the others. The log does not
  time the QR or the collectives separately, so the 4.6 s cannot be split between
  GPU 0's QR (the others wait in the scatter) and the gathers.
- Every QR distance is below `1e-10`. Against the first-run single-GPU time, the
  efficiency on 4 GPUs is 101%; against the 109.9 s run it is 84%, above the 80%
  target. This problem fits on one GPU, so it says little about the reference size.

### D_M = 4000, chi_out = 2000 (reference size)

50 sites, complex128, 184 GB of input on Lustre: 49 environments of 8.19 GB, 401 GB
in all. The single-GPU `run` (job 60143214) read the stack beforehand; the first
`run --devices 4` (job 60137752) did not, so its site stalls include cold reads.
`scaling --devices 4 2` (job 60143212) read the stack beforehand, then ran four and
two GPUs in one process; the QR distance between their outputs is `7.5e-13`.
Speed-up and efficiency are against the single-GPU run.

| GPUs | Time (s) | Speed-up | Efficiency | Left-to-right (s) | Right-to-left (s) | Environment stalls (s) | Site stalls (s) | Environments device / host / disk |
|---|---|---|---|---|---|---|---|---|
| 1 | 3925.3 | 1.00 | 100% | 794.8 | 3112.1 | 2542.3 | 0.0 | 0 / 45 / 4 |
| 2 | 2348.8 | 1.67 | 84% | 729.3-730.6 | 1584.0-1606.3 | 1856.4-1895.6 | 87.1-87.5 | 1 / 35 / 14 |
| 4 | 1099.9 | 3.57 | 89% | 228.0-240.4 | 832.6-844.1 | 697.3-698.5 | 42.1-43.1 | 1 / 45 / 3 |
| 4 (job 60143212) | 1290.4 | 3.04 | 76% | 405.0-407.1 | 855.3-866.8 | 671.1-696.7 | 243.3-243.9 | 2 / 45 / 3 |

Before 5fd4ecf, every run at this size failed with a CuPy out-of-memory error at a
different point (job 59626785 of #59 on `main`, and jobs 59777837 and 59777847
here). The live memory stayed within the plan, 52 GB against 56.7 GB planned, but
the pool held up to 66.7 GB in split blocks and a 12.6-14.7 GB allocation did not
fit under its 67.6 GB cap. Each GPU of every multi-GPU run now split one sketch
batch (192 columns into two pieces) and finished; the single-GPU run did not need to.

Where the time goes:

- Environment stalls are the time a GPU thread waits in the environment store, in
  both passes. The store hands a buffer back only once the GPU has used it, so a
  stall includes waiting for the GPU's own queue, and cannot tell slow transfers
  from a GPU-bound pass; the transfers were not timed separately. They shrink with
  the GPUs: 2542, 1856-1896 and 671-699 s on one, two and four.
- The two 4-GPU runs share the plan but differ by 190 s, and the second has 200 s
  more of site stalls, although it read the stack beforehand and the first did not.
  The cause was not isolated; Lustre is shared, and its bandwidth varies between
  jobs. The efficiency on four GPUs is 76-89% across the two jobs, around the 80%
  target; on two GPUs it is 84%.
- Four GPUs keep only one or two environments on the device, one GPU none. The plan
  picks the largest batches that fit first, and their working set takes each
  GPU's budget (59.0 GB planned peak), although a GPU holds only 2.05 GB of each
  environment.
- Three or four environments go to Lustre on one and four GPUs: the host budget,
  90% of the available memory (about 470 GB), falls short of the 487-496 GB that
  keeping them all in host memory takes. On two GPUs, 14 go to Lustre, with a
  planned host peak of 381.7 GB against 462.6 GB on four; why the plan leaves that
  host memory unused was not investigated.
