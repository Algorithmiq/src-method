## Conclusion

Setup: `O = Z_c`, `O <- U^dag O U` for T = 20 steps, one A100, complex128,
`cutoff=0`, step `t` seeded with `2t` (`2t + 1` for the second half of a pair).
There is one seed per point, so the table has no error bars. The ratios below
hold at every `chi` and both chain lengths, which suggests they are systematic,
but differences under ~1.3x are not individually significant.

**At equal time, pair-trunc wins for thick layers (D = 16), and the stack wins
narrowly for thin ones (D = 4).**

- D = 16, n = 24: pair-trunc reaches 4.8e-4 at 0.31 s per step (`chi = 256`).
  At that time the stack is at `chi` 64-128, with errors of 3.8e-2 to 3.9e-3,
  8-80x worse. To match pair-trunc's error the stack needs `chi = 256` and
  5.1 s per step, 16x longer. n = 12 gives the same picture: 4.5e-4 at 0.11 s
  against 3.5e-4 at 1.7 s.
- D = 4: the stack costs the same as pair-trunc or less (n = 24: 0.05-0.22 s
  against 0.07-0.23 s) and is 1.2-2.1x more accurate. It is therefore the better
  choice at equal time, by a small margin. Oversampling costs 1.5-4x more time
  here, and spending it on a larger `chi` gives lower error: stack at
  `chi = 64` (0.06 s, 2.9e-4) beats stack-os at `chi = 32` (0.12 s, 1.4e-3).

**At equal `chi`, the stack is 1.1-1.3x more accurate than pair-trunc and sits
on pair-wide.** The pair-trunc/stack error ratio is 1.09-1.34 for D = 16, and
1.18-1.34 for D = 4 except at `chi = 16` (1.8-2.1). The stack matches pair-wide
to within 3% for D = 16. Both make a single `chi`-column randomized sketch of
the exact product, so the stack already reaches the pairwise ceiling; pair-wide
pays 3-10x the stack's time (up to 90x pair-trunc's) for nothing more. The only remaining lever is the sketch
itself: `oversample=None` (1.5x) lowers the error another 1.5-3.7x at the same
`chi`.

**Is the stack's accuracy worth its cost? Not for D = 16.** At `chi = 256` the
stack is 1.3x more accurate than pair-trunc for 16x the time and 17x the peak
pool (10.7 GB against 0.62 GB). That ratio exceeds the D/2 = 8 the flop count
predicts. The fixed contraction-path fix (`optimize="dp"`) has not landed (see
Provenance): these timings use opt_einsum's `auto` heuristic, and they look
like it chose poor paths. Stack time grows 7x from `chi` 128 to 256 at n = 24
(5.7x for stack-os), and at n = 12 stack is slower than stack-os at
`chi = 256` (1.73 s against 1.54 s). With better paths the gap should shrink
towards the predicted ~8x, which would still not pay for a 1.3x accuracy gain.
Rerun the D = 16 timings once the fix lands.

**Reference and floor.** U is the same MPO in every method and in both
references, so they share all of the Trotter error and none of it shows up in
these errors. n = 12 errors are dense Frobenius norms. The n = 24 reference is a
gate-by-gate canonical SVD evolution at bond 1024 and cutoff 1e-12. That is the
same exact-SVD truncation as contracting `U^dag O U` and compressing, but done
gate by gate: the whole sandwich would need bond `D^2 chi`, up to 262144. It
agrees with the dense evolution to 8e-12 (n = 8, `checks.json`). At D = 16 it
reaches the 1024 cap from t = 2.6. Its rigorous truncation bound at t = 4 is
1.5e-6, and it differs from the 512 reference by 3.8e-6. The bound is 90x and
the difference 35x below the smallest D = 16 method error (1.3e-4). At D = 4 the bond stays
at 631 or below, so the cutoff alone sets it (bound 6e-10). The n = 24, D = 4,
`chi = 256` rows sit at the inner-product floor (marked `*`): all four methods
are exact there to about 1e-7.
