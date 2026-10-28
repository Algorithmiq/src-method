## Conclusion (stages 1 and 2, and the cost of stage 3)

Heisenberg-picture MPO evolution of Kim et al.'s 127-qubit kicked Ising circuit on
A100-SXM-64GB GPUs, complex128, src-method 0.5.2.dev2 (released code, `dp` paths)
against quimb 1.15 on the same GPU. Per run, one A100 (four independent runs per
debug node, see caveats).

**Bottom line.**

- **Correct.** Every method reproduces dense, statevector, stim and published
  exact references to 1e-13-1e-16 where it should. The weight-10 observable is
  exact to about 1e-14 from `chi = 384`, as in Anand et al.
- **Per compression, src-method's SRC is the best randomized compressor here.**
  Compressing the same exact sandwich at `chi = 256`, it is 2.3x more accurate
  than quimb's SRC in the same time. Oversampled, it is the most accurate method
  measured: relative error 0.050 against zipup 0.056 and fit 0.116.
- **Over a whole evolution, quimb zip-up wins at matched accuracy.** On the
  9-step `<Z62>` and the 5-step weight-17 observables, zip-up (fed left-canonical
  input) reaches a given error at a smaller `chi` and in less time than any
  src-method variant at `theta_h = 8pi/32` and `10pi/32`. src-method beats quimb's
  SRC, which breaks down there, and quimb fit, which is slow and stalls in `chi`.
- **At large `chi` the cost is a host-side SVD, not the GPU.** With a cutoff,
  97% of each compression is NumPy SVDs of the sketch's R factors. Without one,
  the same step runs 4.7x (`chi = 1024`) and 7x (`chi = 2048`) faster, at 6-11
  TFLOP/s per A100. The oversampled variant always pays the host SVD.
- **The 1D MPO does not converge `<Z62>` in the hard regime.** At 9 steps and
  `chi = 1024` the error is 3.6e-3 at `theta_h = 6pi/32`, but 1.6e-2 and 3.0e-2
  at `8pi/32` and `10pi/32`, and not monotonic in `chi` at `8pi/32` (as Anand et
  al. report at 20 steps). Stage 3 can show how far a larger `chi` moves the
  20-step values towards the 2D methods' converged results. It is unlikely to
  match them.

### Decisions that differ from the brief

- **Sub-layers: 6, not 13.** In the qubit-label ordering (Kim et al. SI Fig. S9b,
  the "snake" order Anand et al. use) the busiest cut carries 6 gates, so 6
  sub-layers with one gate per cut is optimal. This matches Kim et al.'s full-layer
  MPO bond of 64 = 2^6. Anand et al. used 13. Every U and U^dag has bond 2. The
  R_X layer is folded exactly into the last sub-layer of each step.
- **Stage 2 angles and depth.** `theta_h = 6, 8, 10 pi/32` (0.589, 0.785, 0.982)
  instead of 0.6, 0.8, 1.0, and `<Z62>` after 9 steps instead of 10. Exact
  references exist only on that grid and at that depth (Begušić, Gray and Chan,
  `reference/bgc/exact.csv`). Tindall's 10-step BP values are not exact.
- **Cutoff 1e-12 for every method in the grid.** Without a cutoff, src-method
  fills every bond to `chi`, whatever the rank of the operator. On a 12-site
  fragment it returned bond 256 for a true-rank-4 sandwich, so the next sandwich
  starts from 4 x 256. quimb's `src` takes its truncation cutoff from
  `project_opts`, and silently uses 1e-10 there. It was given the same 1e-12; the
  default appears as `quimb-src-default`.
- **The exhaustive grid stopped at `chi = 512`** when the plan changed. quimb was
  not run at `chi = 1024`. Seeds: 3 per randomized method; zip-up is
  deterministic.

### Single compression of the same sandwich

`<Z62>` evolved 8 steps at `theta_h = 6pi/32` by src-k1, then sub-layer 0 of step 9
applied exactly (bond up to 4 chi) and compressed back. Five seeds.

| method | chi 64: rel. error | chi 256: rel. error | chi 256: s |
|---|---|---|---|
| optimal SVD truncation | 0.048 | 0.033 | 10.0 |
| src-method | 0.12 | 0.080 | 1.4 |
| src-method, `oversample=None` | 0.075 | 0.050 | 3.4 |
| quimb src | 0.23 | 0.18-0.20 | 1.5 |
| quimb src-oversample | 0.13 | 0.10-0.11 | 3.6 |
| quimb fit (cutoff 1e-12) | 0.066 | 0.116 | 21 |
| quimb zip-up | 0.080 | 0.056 | 6.8 |

### Full evolutions (stage 2, `chi <= 512`, plus src-method at 1024)

Mean absolute error over seeds against the exact value. Time is the whole evolution.

| problem | best src-method | quimb zip-up | quimb fit | quimb src / src-os |
|---|---|---|---|---|
| `<Z62>` 9 steps, `6pi/32` | os, chi 512: 2.5e-3, 355 s | chi 512: 2.2e-3, 186 s | 4.5e-2 at every chi | 1.4-2.7e-2 |
| `<Z62>` 9 steps, `8pi/32` | os, chi 256: 8.9e-3, 110 s | chi 256: 3.8e-3, 86 s | 2.1e-2, 269 s | 0.16 / 0.25 |
| `<Z62>` 9 steps, `10pi/32` | k1, chi 1024: 3.0e-2, 309 s | chi 256: 5.2e-3, 88 s | 1.2e-2, 271 s | 0.12 / 0.11 |
| weight-17, 5 steps, `10pi/32` | k1, chi 1024: 4.0e-4, 122 s | chi 512: 2.4e-4, 72 s | 1.0e-3 at every chi | 5.1e-3 / 3.1e-3 |

- **Fusing sub-layers (k = 2, 3) does not pay.** At equal `chi` the errors match
  k = 1 within seed scatter, and k = 3 costs 2-4x more time and memory.
- **Oversampling** gains about one `chi` doubling in accuracy for 2.5-4x the time.
- **quimb fit with the cutoff stalls:** its error is flat in `chi` and its
  Frobenius norm falls to 0.06-0.1 (0.73 at cutoff 0 in the probe runs).
- **Seed scatter.** src-method's results are tightly clustered (for example
  0.7822-0.7831 over three seeds); quimb's SRC scatters by about 0.02. One quimb
  src seed landing near the exact value is luck, not accuracy.

### Stage 1

- **weight-10:** exact (1e-13-1e-16) from `chi = 384` at every angle.
- **weight-17:** at `chi = 1024`, within 1.2e-4 everywhere except `theta_h =
  10pi/32` and `11pi/32` (4.0e-4 and 9.1e-4). Anand et al. report better than 1e-4
  at their largest `chi`.

### src-method behaviour worth fixing

1. **Bonds padded to `chi_out` when `cutoff = 0`**, whatever the rank (above).
2. **Contraction paths re-planned on jagged bonds.** `_Contractions` is per call
   and keyed on shapes, so an uneven bond profile re-plans `dp` paths site by site.
   At `chi = 256` a cutoff run did 18x fewer flops but took longer (49 s against
   38 s).
3. **Host SVDs.** `truncated_qr` (cutoff > 0) and the oversampled final
   truncation call `np.linalg.svd` on the host, on OpenBLAS here. The comment says
   "device SVDs are slow at this size". Measured (`svd_bench.json`, one square
   complex SVD):

   | n | NumPy gesdd | CuPy gesvd | CuPy gesvdj | GPU eigh(R^dag R) |
   |---|---|---|---|---|
   | 256 | 0.015 s | 0.023 s | 0.11 s | 0.004 s |
   | 1024 | 0.31 s | 0.17 s | 1.2 s | 0.022 s |
   | 2048 | 2.2 s | 0.63 s | 5.2 s | 0.12 s |

   The CPU wins only below `n = 512`; at 2048 the GPU is 3.5x faster, and the
   `eigh` route 19x faster (it squares the condition number, so it only suits the
   truncation to `chi`). The NumPy SVD does not use more than 8 threads.
4. **gesdd fails to converge** on the ill-conditioned R factors:
   `LinAlgError: SVD did not converge`. This happened in 11 of 417 runs, in the
   weight-10 runs when `chi` exceeds the rank and in the oversampled runs at any
   `chi`. The failing matrix is finite, with condition number about 3e20 (a
   127-qubit Pauli string has Frobenius norm 2^63.5), and gesvd converges on it.
   Rerun with a bench-side gesvd fallback, they appear as `+gesvd`. SciPy's gesvd
   is 40x slower than gesdd, so a GPU fallback is the better fix.

### Cost of stage 3 (20-step `<Z62>`, src-k1, cutoff 0)

Measured on one Trotter step of a random operator with the saturated bond profile
(`probe/`). With cutoff 0 every step after the first costs about this much, so the
totals are estimates for a full run at one `theta_h`, not bounds.

| chi | GPUs | s / step | 20 steps | GPU pool / device | host peak |
|---|---|---|---|---|---|
| 1024 | 1 | 66 | 22 min | 9 GB | 17 GB |
| 2048 | 1 | 334 | 1.9 h | 36 GB | 65 GB |
| 2048 | 4 | 223 | 1.2 h | 9.5 GB | 65 GB |
| 4096 | 4 | ~1100 (4 of 6 compressions measured) | ~6 h | 38 GB | 257 GB |
| 8192 | - | - | - | - | ~1 TB: the input and output operators alone exceed a node |

- **Strong scaling at `chi = 2048`** is 1.5x on 4 GPUs. `chi = 4096` runs at
  27 TFLOP/s on 4 GPUs (about 35% of the four GPUs' FP64 peak). Every site stays
  in the device tier.
- **With a cutoff, or oversampled,** the host SVDs make each step 5-13x slower
  (1.5 TFLOP/s) and multi-GPU barely helps: 1.1x on 2 GPUs at `chi = 2048`.
- **A useful stage 3** is `theta_h` in {0.5, 0.7, 0.8, 1.0} at `chi` = 1024, 2048
  and 4096: roughly 30 node-hours on 4-GPU nodes. That needs jobs beyond the
  debug QoS and your approval.

### Caveats

- **Shared nodes.** Grid runs shared a node four ways (one GPU, 8 cores and 118 GB
  each), so host-side work may have contended.
- **Warm-up.** Each method got one untimed warm-up step. The first call of a
  process costs about 20 s for src-method and 3 s for quimb.
- **Environment.** quimb on CuPy 14 needs autoray 0.11, installed in the venv only.
  `prek`'s uv hooks reinstall the locked 0.8.10, which cost the first `chi = 128`
  grid. The job scripts now refuse to start without it.
- **Flops.** Contractions are counted exactly from CuPy's tensordot, einsum and
  matmul; decompositions with nominal LAPACK counts. Products written with `@`
  are not seen.
- **Missing runs.** The oversampled `chi = 1024` runs at `6pi/32` and `10pi/32`
  exceed the 30-minute job limit (stopped at step 8 of 9). quimb fit at
  `chi >= 256` sometimes overran shared job slots.
