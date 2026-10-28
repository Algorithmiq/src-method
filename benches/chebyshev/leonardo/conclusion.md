## Conclusion

Measured on one A100-SXM-64GB per job, complex128. src-method is the released
pipeline with its default `dp` paths, `oversample=1.0`, `cutoff=0`. quimb runs on
the same GPU with its defaults (`cutoff=0` only). Times are medians over steps;
one step is one `apply` and one compression of the three-term sum, and yields two
moments.

**Bottom line.** src-method is the fastest per moment up to `chi = 512`, by
2.5-24x over each quimb method. At matched accuracy it wins only for S(q, w)
errors above about 1e-5; below that, quimb's variational `fit` is faster, because
at equal `chi` it is 35-430x more accurate. The SRC-versus-zip-up gap does not
grow with the MPO bond: it is about 10x for D = 5 and for D = 14. On this
evidence, Chebyshev spectral functions are not yet a clear flagship for SRC. The
case rests on memory (below) and on the sum step, not on raw time to spectrum.
The follow-up below qualifies this: src-method followed by a single fit sweep is
2-3x faster than quimb fit at matched accuracy down to 1e-6. The application was
dropped before that was explored further.

### Time to spectrum at matched accuracy

Chain L = 100, q = 0.990 pi, 200 moments (from the plot and table above):

| target S error | fastest | runner-up |
|---|---|---|
| ~1e-2 | src-method chi 128: 48 s (8.8e-3) | quimb zipup chi 128: 457 s (2.0e-2) |
| ~5e-4 | src-method chi 256: 106 s (5.1e-4) | quimb zipup chi 256: 1104 s (7.5e-4) |
| ~1e-5 | src-method chi 512: 347 s (8.3e-6) | quimb fit chi 128: 351 s (3.2e-5) |
| ~1e-6 | quimb fit chi 256: 483 s (1.2e-6) | src-method needs chi 1024: 1871 s |

YC4 cylinder (D = 14), 50 moments: src-method `chi = 512` reaches 8.3e-5 in 63 s
and quimb fit `chi = 256` 5.6e-5 in 79 s, a near tie. quimb fit `chi = 512`
reaches 1.8e-6 in 189 s; src-method's `chi = 1024` reference takes 330 s.

### Follow-up: oversampling and SRC + one fit sweep (chain, q = pi)

Added after the conclusion above; it changes the matched-accuracy picture.
`src-method+fit1` is src-method followed by one quimb `fit` sweep (warm start,
`max_iterations=1`) on both the apply and the sum; `src-method-os` is
src-method with `oversample=None`. One seed, 100 steps (200 moments).

| method | chi | s / moment | 200 moments | S error |
|---|---|---|---|---|
| src-method+fit1 | 128 | 0.60 | 120 s | 3.6e-5 |
| src-method+fit1 | 256 | 1.14 | 228 s | 1.1e-6 |
| src-method+fit1 | 512 | 3.41 | 681 s | 1.5e-7 |
| src-method-os | 128 | 1.20 | 239 s | 4.7e-5 |
| src-method-os | 256 | 3.69 | 738 s | 1.3e-6 |
| src-method-os | 512 | 16.5 | (4 steps, stopped) | - |
| quimb fit | 128 | 1.76 | 351 s | 3.2e-5 |
| quimb fit | 256 | 2.42 | 483 s | 1.2e-6 |

- The hybrid is 2.1-2.9x faster than quimb fit at matched accuracy and beats
  plain src-method at every accuracy measured; at L = 20 it matches fit's
  accuracy per `chi`.
- `oversample=None` buys the same accuracy per `chi` as the fit sweep but costs
  2-3x more than the hybrid, and 5-10x more than plain src-method.
- 1.5e-7 is below the reference's resolution (src-method `chi = 1024`); it shows
  agreement, not a measured error.
- The capacity probes (largest `chi` per method on one card; src-method on 4
  GPUs) were cancelled before running when the application was dropped.

### Speed per moment and the D scaling

| s / moment | chi 128 | chi 256 | chi 512 | chi 1024 |
|---|---|---|---|---|
| chain D = 5, src-method | 0.24 | 0.53 | 1.73 | 9.4 |
| chain, quimb zipup | 2.29 | 5.52 | 15.9 | 40.4 (OOM at step 3) |
| chain, quimb src | 1.03 | 2.15 | 4.33 | **7.2** |
| chain, quimb fit | 1.76 | 2.42 | 5.29 | 17.2 |
| cylinder D = 14, src-method | | 0.35 | 1.27 | 6.6 |
| cylinder, quimb zipup | | 3.51 | 10.5 | 35.2 |
| cylinder, quimb src | | 1.34 | 3.25 | 9.6 |
| cylinder, quimb fit | | 1.58 | 3.78 | 13.8 |

- **zipup / src-method:** 9.6, 10.5, 9.1 and 4.3 on the chain; 10.1, 8.3 and
  5.3 on the cylinder. The ratio does not grow with D, and it shrinks with `chi`.
- **SRC loses at `chi = 1024` on the chain:** quimb's own SRC takes 7.2 s per
  moment against 9.4 s for src-method. src-method's time is dominated by the
  compression of the sum (bond `3 chi`): 13.7 s of compress against 5.0 s of
  apply. Its lead over quimb src falls from 4.3x at `chi = 128` to 0.77x.
- **quimb zipup-first** (oversampled zip-up) is accurate but the slowest method,
  at 24-28x src-method.

### Memory

Peak CuPy pool at `chi = 1024`: src-method 18.5 GB (chain) and 14.9 GB
(cylinder); quimb src 10 GB and 39 GB; quimb fit 56 GB and 40 GB. Both quimb
zip-ups exceed the 64 GB card on the chain, in the sum compression at step 3
(zipup at 52 GB, zipup-first at 61 GB). src-method is the only method below
20 GB on both models.

### Accuracy

- **Phase 1 (L = 20, against exact sparse moments, 200 moments):** src-method,
  quimb src, zipup (with left-canonical input) and fit all converge with `chi`;
  the worst error / mu_0 at `chi = 256` is 7e-7, 1e-6, 2e-6 and 3e-9.
- **Equal chi:** quimb fit is 35-430x more accurate than src-method in S(q, w),
  and zipup-first 7-190x. Without oversampling, SRC pays for its speed in accuracy
  per `chi`; `oversample` was left at its default, as the brief specifies.
- **quimb src at L = 100, q = pi** degrades badly (moment error 38 and 14 at
  `chi = 128, 256`; S error 0.46 and 0.09), although it matches src-method at
  L = 20 and on the cylinder. Not diagnosed.
- **Reference:** src-method `chi = 1024`, seed 0. Against `chi = 512` it
  differs by 6e-6 (chain, q = pi), 1e-5 (q = pi/2) and 8e-5 (cylinder) in S(q, w).
  quimb fit at `chi = 512` and `1024` lands 2-3e-6 from it on the cylinder, so
  errors near 1e-6 are at the reference's own resolution.
- **Physics check:** 98.4% (q = pi/2) and 100% (q = pi) of the reference weight
  lies between the des Cloizeaux-Pearson and two-spinon edges, widened by the
  kernel width. That width, pi a / N = 0.55 for 200 moments (the full bandwidth
  is about 69), is too coarse to resolve the continuum. Energy truncation, or
  many more moments, is needed for useful lineshapes; on the cylinder, 50 moments
  give a width of 2.4.

### Findings about the stack (not about SRC)

- **autoray:** quimb 1.15 on CuPy 14 needs autoray 0.11 (installed in the venv
  only). autoray 0.8.10 uses the unhashable CuPy `Device` as a cache key and
  fails in every decomposition.
- **quimb on cupy:** quimb's `src` noise and `fit` initial guess are drawn with
  numpy whatever the backend; a shim moves them to the device.
- **quimb `apply_op_vec`:** `tensor_network_apply_op_vec(..., compress=True,
  method=...)` on an MPS calls `MPS.compress`, whose `method` is the
  decomposition. zipup, src and fit need a lazy apply followed by
  `tensor_network_1d_compress`.
- **quimb `zipup` is not exact at full bond** (`diagnose_zipup.json`: 14-60% on
  a 12-site state at `chi = 64`, in any gauge). Fed right-canonical input, its
  L = 20 moments diverge (`quimb-zipup-rightcanon` rows: errors 0.5-13).
  Left-canonicalising the input, timed and used in every reported zipup run,
  makes them converge.
- **DMRG top state:** from a random start, DMRG on -H crept towards the
  ferromagnetic multiplet (a Dicke state of bond n/2 + 1) and had not converged
  after 14 sweeps on 100 sites. The chain's top-state DMRG starts from
  |up...up>; the cylinder's converged from a random start.

### Coverage and caveats

- **src-method:** 3 seeds at `chi <= 512` on the chain, both q, 100 steps;
  1 seed at `chi = 1024`; 1 seed on the cylinder.
- **quimb:** q = pi only, one seed. 100 steps at `chi <= 256`. Timing-only
  (15, 8, 2-4 steps) at `chi = 512, 1024`, and for zipup-first at `chi = 512`
  on the cylinder (9 steps). zipup-first on the cylinder at `chi = 1024` was not
  run.
- **Race:** two jobs once ran the same cylinder `chi = 1024` configurations at
  once and corrupted their files. Both were cancelled, the files deleted and the
  runs redone in one job; the reported timings agree with both overlapping runs
  (quimb src 19.3 s per step).
- **Not done:** the optional TDVP baseline.
