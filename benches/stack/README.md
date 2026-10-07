# Stack depth benchmarks

`bench_depth.py` compares one-shot SRC over a stack (`src(A_1, ..., A_k, psi)`)
with sequential pairwise application (`apply` from right to left, truncating to
`chi` after every product), as a function of the number of trains in the stack.
At depth 2 both methods run the same computation, which makes that row a sanity
check.

- `accuracy`: median relative error against the exact dense product on 10-site
  chains over 20 random instances, next to `lower_bound`, the largest
  best-rank-`chi` tail over all cuts, which no train of bond dimension `chi` can
  beat. `one_shot_wins` is the fraction of instances where one-shot is more
  accurate, ignoring ties and instances where both are exact (`nan` if none is
  left).
- `timing`: best-of-3 wall time on 30-site chains, MPS bond 64, `chi = 64`.
  `ratio` is one-shot time over sequential time.

Families: `random` are complex Gaussian MPOs of bond 3 (flat spectra);
`trotter-<dt>` are brickwork layers of a mixed-field Ising model with random
couplings (bond at most 4, decaying spectra). A `.mps` suffix ends the stack with
an MPS. In `timing`, `x<k> . mps` is `k` layers applied to the MPS.

```bash
uv run python benches/stack/bench_depth.py accuracy --output accuracy.md
uv run python benches/stack/bench_depth.py timing --output timing.md
```

Tuple options repeat the flag, e.g. `--depths 2 --depths 3`.

## Results

Measured on an AMD Ryzen 7 7840U, 16 threads, with light background load.

### Accuracy

| family | depth | chi | lower_bound | one_shot | sequential | one_shot_wins |
|---|---|---|---|---|---|---|
| random.mps | 2 | 4 | 0.168 | 0.499 | 0.499 | nan |
| random.mps | 2 | 8 | 0.0298 | 0.0989 | 0.0989 | nan |
| random.mps | 2 | 16 | 2.59e-16 | 1.58e-15 | 1.58e-15 | nan |
| random.mps | 3 | 4 | 0.194 | 0.519 | 0.636 | 0.9 |
| random.mps | 3 | 8 | 0.0584 | 0.188 | 0.191 | 0.75 |
| random.mps | 3 | 16 | 0.00667 | 0.0196 | 0.0194 | 0.35 |
| random.mps | 4 | 4 | 0.263 | 0.611 | 0.773 | 0.9 |
| random.mps | 4 | 8 | 0.0978 | 0.275 | 0.335 | 0.85 |
| random.mps | 4 | 16 | 0.0145 | 0.041 | 0.0533 | 0.8 |
| random | 2 | 8 | 0.0734 | 0.318 | 0.318 | nan |
| random | 2 | 16 | 1.3e-15 | 1.53e-15 | 1.53e-15 | nan |
| random | 2 | 32 | 7.34e-16 | 1.61e-15 | 1.61e-15 | nan |
| random | 3 | 8 | 0.289 | 0.75 | 0.755 | 0.7 |
| random | 3 | 16 | 0.102 | 0.352 | 0.351 | 0.65 |
| random | 3 | 32 | 9.17e-16 | 2.19e-15 | 2.7e-15 | nan |
| random | 4 | 8 | 0.397 | 0.828 | 0.905 | 1 |
| random | 4 | 16 | 0.217 | 0.579 | 0.607 | 0.95 |
| random | 4 | 32 | 0.0819 | 0.274 | 0.27 | 0.55 |
| trotter-0.3.mps | 2 | 2 | 0.124 | 0.265 | 0.265 | nan |
| trotter-0.3.mps | 2 | 4 | 6.04e-16 | 1.28e-15 | 1.28e-15 | nan |
| trotter-0.3.mps | 2 | 8 | 3.4e-16 | 1.38e-15 | 1.38e-15 | nan |
| trotter-0.3.mps | 3 | 2 | 0.124 | 0.388 | 0.432 | 0.65 |
| trotter-0.3.mps | 3 | 4 | 0.000264 | 0.00162 | 0.00125 | 0.5 |
| trotter-0.3.mps | 3 | 8 | 5.53e-16 | 1.41e-15 | 1.68e-15 | nan |
| trotter-0.3.mps | 4 | 2 | 0.15 | 0.502 | 0.517 | 0.45 |
| trotter-0.3.mps | 4 | 4 | 0.0023 | 0.0119 | 0.0103 | 0.4 |
| trotter-0.3.mps | 4 | 8 | 2.57e-08 | 1.16e-07 | 1.14e-07 | 0.5 |
| trotter-0.3 | 2 | 4 | 2.87e-15 | 1.27e-15 | 1.27e-15 | nan |
| trotter-0.3 | 2 | 8 | 2.2e-15 | 1.11e-15 | 1.11e-15 | nan |
| trotter-0.3 | 2 | 16 | 1.38e-15 | 1.2e-15 | 1.2e-15 | nan |
| trotter-0.3 | 3 | 4 | 0.000457 | 0.00324 | 0.00241 | 0.5 |
| trotter-0.3 | 3 | 8 | 2.17e-07 | 2.65e-06 | 1.76e-06 | 0.35 |
| trotter-0.3 | 3 | 16 | 1.95e-15 | 1.47e-15 | 1.65e-15 | nan |
| trotter-0.3 | 4 | 4 | 0.000522 | 0.00471 | 0.00487 | 0.55 |
| trotter-0.3 | 4 | 8 | 2.45e-07 | 3.78e-06 | 4.85e-06 | 0.6 |
| trotter-0.3 | 4 | 16 | 1.98e-15 | 1.75e-15 | 2.4e-15 | nan |
| trotter-0.8.mps | 2 | 4 | 6.44e-16 | 1.33e-15 | 1.33e-15 | nan |
| trotter-0.8.mps | 2 | 8 | 4.41e-16 | 1.36e-15 | 1.36e-15 | nan |
| trotter-0.8.mps | 2 | 16 | 1.93e-16 | 1.23e-15 | 1.23e-15 | nan |
| trotter-0.8.mps | 3 | 4 | 0.00926 | 0.0393 | 0.0393 | 0.5 |
| trotter-0.8.mps | 3 | 8 | 6.2e-16 | 1.45e-15 | 1.85e-15 | nan |
| trotter-0.8.mps | 3 | 16 | 2.97e-16 | 1.53e-15 | 1.65e-15 | nan |
| trotter-0.8.mps | 4 | 4 | 0.0647 | 0.223 | 0.202 | 0.4 |
| trotter-0.8.mps | 4 | 8 | 7.87e-05 | 0.000315 | 0.00041 | 0.55 |
| trotter-0.8.mps | 4 | 16 | 4.6e-16 | 1.76e-15 | 2.27e-15 | nan |
| trotter-0.8 | 2 | 8 | 1.79e-15 | 1.29e-15 | 1.29e-15 | nan |
| trotter-0.8 | 2 | 16 | 1.19e-15 | 1.4e-15 | 1.4e-15 | nan |
| trotter-0.8 | 2 | 32 | 6.45e-16 | 1.21e-15 | 1.21e-15 | nan |
| trotter-0.8 | 3 | 8 | 0.000394 | 0.00364 | 0.00357 | 0.5 |
| trotter-0.8 | 3 | 16 | 1.78e-15 | 1.59e-15 | 1.97e-15 | nan |
| trotter-0.8 | 3 | 32 | 8.17e-16 | 1.7e-15 | 1.91e-15 | nan |
| trotter-0.8 | 4 | 8 | 0.00048 | 0.00582 | 0.00579 | 0.55 |
| trotter-0.8 | 4 | 16 | 1.73e-15 | 2.81e-15 | 4.16e-15 | nan |
| trotter-0.8 | 4 | 32 | 8.41e-16 | 1.88e-15 | 2.63e-15 | nan |

### Timing

| family | depth | chi | one_shot_s | sequential_s | ratio |
|---|---|---|---|---|---|
| trotter-0.3 x1 . mps | 2 | 64 | 0.0764 | 0.0642 | 1.19 |
| trotter-0.3 x2 . mps | 3 | 64 | 0.0936 | 0.147 | 0.639 |
| trotter-0.3 x3 . mps | 4 | 64 | 0.201 | 0.207 | 0.975 |
| trotter-0.3 x4 . mps | 5 | 64 | 0.384 | 0.283 | 1.36 |
| random D=8 x1 . mps | 2 | 64 | 0.283 | 0.28 | 1.01 |
| random D=8 x2 . mps | 3 | 64 | 1.67 | 0.52 | 3.21 |
| random D=8 x3 . mps | 4 | 64 | 24.7 | 0.837 | 29.5 |
| random D=16 x1 . mps | 2 | 64 | 0.638 | 0.622 | 1.03 |
| random D=16 x2 . mps | 3 | 64 | 11.8 | 1.31 | 9.05 |
| random D=16 x3 . mps | 4 | 64 | 316 | 2.33 | 136 |

## Summary

- One sweep over the whole stack is at best moderately more accurate than
  pairwise application. For random stacks ending in an MPS it has about 20 %
  lower median error at depth 4 for every `chi` measured, and at depth 3 only
  at the smallest `chi` (at larger `chi` the gain vanishes); it wins 75-90 % of
  those instances. For random MPO products the gain is at most 8 % (depth 4,
  small `chi`). For Trotter layers there is no systematic difference (wins
  35-65 %).
- Both methods sit 2-15x above the lower bound, so the sketch, not compounding
  across layers, dominates the error. Oversampling
  ([#40](https://github.com/Algorithmiq/src-method/issues/40)) is the larger
  lever.
- The per-site cost grows with `chi**2` times the product of the layer bonds.
  Thin Trotter layers run at 0.6-1.4x the sequential time up to depth 5, while
  random MPOs of bond 8 and 16 at depth 4 are 30x and 136x slower. Stack
  shallow, thin layers; apply anything else pairwise.
