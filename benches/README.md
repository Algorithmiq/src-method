# SRC benchmarks

This directory collects the experiments used to measure the performance of the
different SRC variants. They are grouped by the kind of workload exercised:

- [`primitives/`](primitives/) — Synthetic micro-benchmarks of the four core
  SRC primitives (MPO–MPO, MPO–MPS, …). Used to assess speedup and accuracy of
  individual primitives versus `quimb` references on a fixed problem size.
  See [`primitives/README.md`](primitives/README.md).
- [`stack/`](stack/) — One-shot SRC over stacks of trains against sequential
  pairwise application, by stack depth: accuracy against dense references and
  wall time. See [`stack/README.md`](stack/README.md).
- [`large/`](large/) — Out-of-core SRC of `N . V . M . U` with a large `M` read
  from disk, on one GPU: plan, wall time, memory peaks and stall time. See
  [`large/README.md`](large/README.md).
