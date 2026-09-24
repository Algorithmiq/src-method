# Design: one SRC kernel for arbitrary-depth stacks

- Issue: [#34](https://github.com/Algorithmiq/src-method/issues/34)
- Related: [#40](https://github.com/Algorithmiq/src-method/issues/40) (oversampling, out of scope)
- Status: approved in discussion, pending spec review

## Goal

Replace the four hand-written SRC kernels (`_src_mps`, `_src_mpo`, `_src_mpo_mps`,
`_src_mpo_mpo`) with one generic kernel that contracts and compresses a *stack* of
trains of any depth, exposed through a new public `src`. `apply` and `compress`
keep their signatures and become thin wrappers.

This is primarily a refactor. The k = 3 spike showed that compressing a stack in one
shot is at best modestly more accurate than applying it pairwise, and much more
expensive once MPO bonds are not thin (see [Spike findings](#spike-findings)).

## Stacks and conventions

A stack is an ordered sequence of trains `T_1 . T_2 . ... . T_m` contracted along
their physical legs. The trains follow the `quimb` layouts already used by the
package: MPO bulk `(l, r, u, d)`, MPS bulk `(l, r, p)`, boundaries drop the outer bond.

Well-formed stacks:

| Stack | Meaning | Output |
|---|---|---|
| `[A]` | compress an MPO | MPO |
| `[psi]` | compress an MPS | MPS |
| `[A_1, ..., A_k]` | `A_1 A_2 ... A_k` | MPO |
| `[A_1, ..., A_k, psi]` | `A_1 ... A_k psi` (ket) | MPS |
| `[phi, A_1, ..., A_k]` | `phi^T A_1 ... A_k` (bra) | MPS |

- Stack order is mathematical order: in `src(A, B, psi)`, `B` acts first, matching
  `apply(left, right)`.
- Contraction always joins the `d` leg of a train with the `u` (or `p`) leg of the
  train to its right. A leading MPS joins its `p` leg with the `u` leg of `A_1`; the
  output leg is `A_k.d`.
- **No conjugation.** A leading MPS is a row vector used as-is:
  `v[d] = sum phi[u_1] A_1[u_1, d_1] ... A_k[d_{k-1}, d]`. For the physical bra
  `<psi| A_1 ... A_k`, pass `psi.conj()`; the result then pairs with a ket by plain
  contraction. This keeps every stack linear in every input, avoids silent double
  conjugation, matches `quimb` (conjugation is always explicit), and makes the bra
  case a relabelling of the ket case with no new kernel:
  `[phi, A_1, ..., A_k] == [A_k^T, ..., A_1^T, phi]`, where `A^T` swaps `u` and `d`.
- Position decides role: an MPS first is a bra, an MPS last is a ket, a single MPS
  is a compression. An MPS in the middle, or at both ends of a stack of two or more
  trains (a scalar), is rejected.

## Public API

```python
def src(
    *trains: Sequence[NDArray],
    chi_out: int,
    cutoff: float = 0.0,
    dtype: type = np.float64,
    seed: int | None = None,
    device: str = "cpu",
) -> list[NDArray]:
```

- `chi_out` and the options are keyword-only and mean what they mean in `apply`
  today.
- Returns an MPS if either end of the stack is an MPS, otherwise an MPO, in
  right-canonical form, as numpy arrays.
- Exported from `src_method.__init__`. The name is provisional; the module is called
  `stack.py` so it survives a rename and does not collide with the function.
- `apply(left, right, chi_out, ...)` keeps its current type check (MPO on the left)
  and forwards to `src(left, right, ...)`; the bra form is reachable only through
  `src`. `compress(train, chi_out, ...)` forwards to `src(train, ...)`.

## Module layout

| File | Contents |
|---|---|
| `stack.py` (new) | Public `src()`: normalize and validate, then dispatch to the exact fallback or the sweep; logging. |
| `_sweep.py` (new) | Private generic kernel on a ket-form stack (MPOs with an optional trailing MPS): padding, generated equations, per-call expression cache. |
| `_tensor_train.py` | Adds `pad`, `unpad`, a `u <-> d` transpose, `normalize_stack(trains) -> (layers, out_kind)` (validation plus bra-to-ket rewrite), and `exact_stack` (generalizes `exact_apply`). `exact_compress`, `infer_kind`, `check_exact_supported` stay. |
| `apply.py`, `compress.py` | Wrappers only. The four `_src_*` kernels and the duplicated log strings are removed. |

## Validation

Performed in `normalize_stack`, before any logging:

| Condition | Error |
|---|---|
| No trains | `ValueError` |
| A train whose first site is neither rank 2 nor rank 3 | `TypeError` |
| MPS in the middle, or at both ends of a stack of two or more trains | `TypeError` |
| Trains with different site counts | `ValueError` |
| Adjacent layers whose joined physical legs differ in size at some site | `ValueError` naming the site and the two layers (new) |
| One site | `ValueError`, raised before the fallback warning |
| Two sites | warning, then `exact_stack` |

`chi_out` and bond consistency within a train are not validated, as today.

## Sweep

On a ket-form stack of `k` layers and `n >= 3` sites:

1. Transfer every array to the device once (`xp.asarray`), then pad by views: a
   boundary site gains a size-1 outer bond, an MPS gains a size-1 `d` leg. Every site
   of every layer is then `(l, r, u, d)`.
2. Obtain the four einsum equations for depth `k` from `_equations(k)`, a pure
   function behind `functools.lru_cache` keyed on `k` only:
   - left-to-right: `C[s, a_1..a_k], omega[s, u, d], layers -> C[s, b_1..b_k]`
   - right-to-left `M`: `C[s, a..], layers, S[r, b..] -> M[r, u, d, s]`
   - right-to-left `S`: `eta*[l, r, u, d], layers, S[r, b..] -> S[l, a..]`
   - first site: `layers, S[r, b..] -> [a.., r, u, d]`

   The sketch index `s` is shared by `C`, `omega` and the output (Khatri-Rao
   sketch), as in the current kernels.
3. Left to right: `C_0 = ones((chi_out,) + (1,) * k)`. For sites `0 .. n-2` draw
   `omega` of shape `(chi_out, u, d)` (`d = 1` for MPS output) and form `C_{j+1}`.
   `omega` stays a real Gaussian drawn on the host from `default_rng(seed)` and cast
   to `dtype`, so a seed gives the same draws on CPU and GPU.
4. Right to left: `S = ones((1,) + (1,) * k)`. For sites `n-1 .. 1`: form `M`, take
   `Q = truncated_qr(M.reshape(r * u * d, chi_out), cutoff, xp)`, reshape to
   `eta_j` of shape `(rank, r, u, d)`, update `S` with `eta_j.conj()`, free `C_j`.
   Starting from `S = ones` removes the last-site special case. Site 0 absorbs `S`.
   `eta_1 .. eta_{n-1}` are right isometries, so the output is right-canonical.
5. Unpad and `to_numpy`.

Expression cache: a dictionary per call, keyed on `(equation, operand shapes)`,
holding `opt_einsum.contract_expression` objects called on the device arrays
(`opt_einsum` dispatches to CuPy). Uniform bulk sites share one entry; jagged bonds
add entries; nothing persists across calls. The cache is kept only if it measures
faster than plain `contract` on the existing MPO-MPO benchmark.

Logging (structlog): info "Starting SRC" with `n_sites`, `depth`, `output` and
`device`; debug the estimated largest environment size,
`chi_out * max over cuts of prod(bond dims)`, in elements; debug sweep timings; the
existing small-system warning.

No cost guard beyond the debug log and documentation.

## Exact fallback

`exact_stack(layers, chi_out, out_kind)` for two-site stacks, on the host: at each of
the two sites fold the layers pairwise on padded tensors,
`(l1, r1, u, x) . (l2, r2, x, d) -> (l1 l2, r1 r2, u, d)`, until one layer remains;
unpad; call `exact_compress`. The bra form reaches it already rewritten to ket form.

## Documentation

- `README.md` and `docs/index.md`:
  - feature list gains arbitrary-depth stacks through `src`, with a usage snippet;
  - new "Contraction conventions" subsection under "Tensor Indexing Conventions":
    the `d <-> u` table, stack order, role by position, the no-conjugation rule for a
    leading MPS with the `.conj()` recipe and the transpose identity;
  - cost note: per-site cost grows with `chi_out^2 * prod(D_i)`; stack shallow, thin
    layers (e.g. two or three Trotter layers) and apply anything else pairwise; link
    to the depth benchmark.
- `src` docstring carries the conventions and the cost note in brief; `apply` and
  `compress` docstrings point to `src`.

## Testing

New `tests/test_stack.py`; `tests/test_package.py` stays unchanged as the wrapper
regression suite.

- Exactness against a dense reference (5-6 sites, `chi_out` large enough to avoid
  truncation, complex inputs, jagged bonds) for `[A]`, `[psi]`, `[A, psi]`, `[A, B]`,
  `[A, B, psi]`, `[A, B, C]`, `[A, B, C, psi]`, `[psi, A]`, `[psi, A, B]`. The dense
  helper lives in the test file.
- Convention pin: `src(psi, A)` matches `psi^T A` and is far from `psi^dagger A`.
- Identity: `src(I, I, A)` reproduces `A` with `chi_out` equal to `A`'s bond.
- Wrappers: `apply` and `compress` equal `src` bit-for-bit for the same seed.
- Validation: one test per row of the validation table, with `match=`.
- Two-site stacks of depth 3, bra included, with the fallback message checked via
  `caplog`.
- `cutoff` on a depth-3 stack; seed determinism.
- CPU/GPU parity for a depth-3 stack in `tests/test_gpu_backend.py`.
- `perf` benchmark of a depth-3 Trotter-like stack next to the MPO-MPO one.

## Benchmarks

`benches/stack/bench_depth.py`, a `cyclopts` CLI, measuring as a function of depth:

- accuracy against a dense reference on small chains, with the cut lower bound
  (largest best-rank-`chi` tail over all cuts), for random and Trotter-like layers;
- wall time of one-shot `src` against sequential pairwise `apply` at realistic size.

Measured tables go in `benches/stack/README.md`, linked from `benches/README.md` and
from the cost note in the docs.

## Delivery

- One PR closing #34, its description carrying the spike findings (which answer the
  issue's open questions) and a link to #40.
- Commit sequence: `refactor` (generic sweep, wrappers, old kernels removed),
  `feat(stack)` (public `src`), `docs`, `test`, `perf` (benchmarks).
- `feat` implies a minor version bump. Not breaking: signatures are unchanged. Seeded
  results change in the last digits (MPO-MPS drew `omega` as `(phys, chi)`, the
  generic kernel as `(chi, phys)`, and contraction order changes rounding); seeds stay
  reproducible within a version. Stated in the PR description.
- `uv run prek run --all-files` and the full test suite pass before pushing.

## Out of scope

- Oversampling and an exact final truncation (#40).
- A hyper-optimizing contraction path backend such as `cotengra`: the spike showed
  `opt_einsum`'s default within 1.5x of optimal at the depths where one-shot
  compression is worthwhile.
- A better public name than `src`.

## Spike findings

Throwaway code, dense references on 10-site chains, 20 seeds, medians.

- Correctness: the padded generic sweep reproduces the exact product to about 1e-15
  for `A.psi`, `A.B`, `A.B.psi`, `A.B.C`, `A.B.C.psi` and `psi.A.B`.
- Performance parity for today's cases: generic kernel between 0.91x and 1.11x the
  time of the hand-written kernels (MPS, MPO, MPO-MPS, MPO-MPO; 10-30 sites).
- Accuracy, one-shot against sequential at equal `chi_out`: up to about 35 % lower
  error for random MPO^k . MPS (flat spectra, one-shot wins 55-100 % of seeds, more
  often at smaller `chi_out` and greater depth); no systematic difference for random
  MPO^k or Trotter layers.
- Both are 3-10x above the cut lower bound. Sketching at `2 chi` and truncating
  exactly brings both within 1-1.5x of the bound (#40).
- Cost (30 sites, MPS bond 64, `chi_out` 64), one-shot relative to sequential:
  Trotter layers 0.8x at k = 2 and 1.2-1.5x at k = 3-4; random MPO bond 8 at k = 3,
  27x; bond 16 at k = 3, 170x. The cost is intrinsic (`chi^2 * prod(D_i)` per site),
  not a path-finding problem.
