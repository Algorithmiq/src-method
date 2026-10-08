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
