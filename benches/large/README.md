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

Pending: `scaling --devices 4 2` (job 59777837) and the single-GPU `run` for `T_1`
(job 59777847) are queued on the default QoS, both with the stack read beforehand.
