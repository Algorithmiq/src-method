# SRC benchmarks

This directory collects the experiments used to measure the performance of the
different SRC variants. They are grouped by the kind of workload exercised:

- [`primitives/`](primitives/) — Synthetic micro-benchmarks of the four core
  SRC primitives (MPO–MPO, MPO–MPS, …). Used to assess speedup and accuracy of
  individual primitives versus `quimb` references on a fixed problem size.
  See [`primitives/README.md`](primitives/README.md).
- [`stack/`](stack/) — One-shot SRC over stacks of trains against sequential
  pairwise application, by stack depth: accuracy against dense references and
  wall time on small chains, and quench dynamics of 50 and 100 qubits on the GPU.
  See [`stack/README.md`](stack/README.md).
- [`large/`](large/) — Out-of-core SRC of `N . V . M . U` with a large `M` read
  from disk, on one GPU: plan, wall time, memory peaks and stall time. See
  [`large/README.md`](large/README.md).

The scripts need the `bench` dependency group (included in `dev`):

```bash
uv sync --group bench
```

## Leonardo

Each folder has a `leonardo/` subfolder with a Slurm script for the Booster
partition of [Leonardo](https://docs.hpc.cineca.it/general/getting_started.html)
at CINECA (one node: 32 Xeon 8358 cores, 4 A100-64GB, 512 GB of memory), and the
logs of the recorded runs. Set up the environment once, from the repository
root on a login node:

```bash
# Keep the uv cache and Python installs out of $HOME.
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv sync --all-groups --extra gpu-nvidia

# The Booster driver (535) predates CUDA 13: unpack NVIDIA's forward-compat libcuda.
mkdir -p .cuda-compat && cd .cuda-compat
curl -sSfO https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-compat-13-4-615.71.09-1.el8.x86_64.rpm
rpm2cpio cuda-compat-13-4-*.rpm | cpio -idm && cd ..
```

The scripts prepend `.cuda-compat/usr/local/cuda-13.4/compat` to
`LD_LIBRARY_PATH` (override with `CUDA_COMPAT`) and keep CuPy's kernel cache in
`.cupy-cache/`. Submit from the `leonardo/` folder, passing the benchmark's own
arguments, e.g. `sbatch run.sh --n-sites 25 --chi-out 1000 --device gpu`. The
default QoS allows long runs but can queue for hours; for jobs under 30 minutes
add `--qos=boost_qos_dbg --time=00:30:00`, which starts quickly but takes at
most two jobs per user.
