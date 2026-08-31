# SRC benchmarks

This directory collects the experiments used to measure the performance of the
different SRC variants. They are grouped by the kind of workload exercised:

- [`primitives/`](primitives/) — Synthetic micro-benchmarks of the four core
  SRC primitives (MPO–MPO, MPO–MPS, …). Used to assess speedup and accuracy of
  individual primitives versus `quimb` references on a fixed problem size.
  See [`primitives/README.md`](primitives/README.md).
