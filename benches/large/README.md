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
