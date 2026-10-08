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
