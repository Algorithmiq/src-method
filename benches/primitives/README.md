# SRC primitive benchmarks

This directory contains synthetic micro-benchmarks of the SRC primitives,
run on multiple supercomputing systems. They isolate the cost of a single
primitive (e.g. MPO–MPO contract+compress) on randomly generated inputs of
controlled bond dimension, and compare to a `quimb` reference.

## Leonardo

The Leonardo [supercomputer](https://docs.hpc.cineca.it/general/getting_started.html) at CINECA is
a pre-exascale Tier-0 EuroHPC system, the 10th fastest in the world as of June 2025. In the
corresponding folder you can find the scripts to run the benchmarks on it, and some example
results. It is assumed that the user has built the `uv` virtual environment in the root
of the repository. This can be done by running:

```bash
uv sync --all-groups --all-extras
```

### MPO-MPO benchmark

The benchmark contracts two length-`n_sites` matrix product operators (MPOs) whose initial bond dimension is `chi_out` (unless `chi_id` is specified for the second MPO), then compresses the resulting MPO back to bond dimension `chi_out`. One MPO is fully random; the other is an identity MPO perturbed by summing it with another random MPO with small elements, making the contraction highly compressible. The table below compares the performance of `quimb` versus `src` in terms of CPU time, total memory usage (maximum resident set size, which is the peak memory usage of the job), relative speedup, and accuracy.

| Experiment        | CPUs | Library       | Time (s) | MaxRSS (GB) | Speedup | Distance to reference |
|-------------------|------|---------------|----------|-------------|---------|-----------------------|
| 50 sites chi=50   | 32   | quimb         | 164.97   |             |         | 0.0                   |
|                   |      | src           | 11.92    | 38.79       | 13.8x   | 2.1e-08               |
| 50 sites chi=50   | 112  | quimb         | 238.67   | 38.87       |         | 2.98e-08              |
|                   |      | src           | 18.88    | 38.33       | 12.6x   | 2.98e-08              |
| 20 sites chi=100  | 32   | quimb         | 2004.52  |             |         | 0.0                   |
|                   |      | src           | 94.83    | 240.79      | 21x     | 2.6e-08               |
| 25 sites chi=1000 | 112  | quimb (svd)   | 564.34   | 23.98       |         |                       |
|          chi_id=4 |      | quimb (rsvd)  | 501.82   | 24.10       |         |                       |
|                   |      | src           | 71.01    | 4.03        | 7x      |                       |
| 50 sites chi=1000 | 112  | quimb (rsvd)  | 1187.82  | 49.47       |         |                       |
|          chi_id=4 |      | src           | 150.11   | 7.01        | 8x      |                       |

### MPO-MPO strong scaling

For this benchmark, we fix the problem size to `n_sites=25`, `chi=1000` and `chi_id=4`, and vary the number of CPUs in a single node. We tweak the OpenMP environment variables in the `run.sh` script to optimize performance, for instance `OMP_PLACES` and `OMP_PROC_BIND`, so the NUMA domain is taken into account. We do this both manually and automatically.

| CPUs | Time auto (s)   | Time maual (s)|
|------|-----------------|---------------|
| 2    | 135.36          | 135.74        |
| 4    | 93.61           | 94.06         |
| 8    | 72.84           | 72.86         |
| 14   | 64.92           | 64.84         |
| 28   | 63.63           | 63.84         |
| 56   | 68.53           | 68.60         |
| 112  | 71.57           | 70.74         |

So the difference between manual and automatic binding is negligible, and the best performance is
achieved with 28 CPUs. Beyond that, the performance degrades, likely due to memory bandwidth limits or NUMA overhead.
