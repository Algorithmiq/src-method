# Chebyshev MPS spectral functions: src-method against quimb

## Provenance

src-method 0.5.2.dev3+gf8d0f6f76, quimb 1.15.0, CuPy 14.2.0, NumPy 2.4.4, CUDA runtime 13020, driver 13040, NVIDIA A100-SXM-64GB, TeNPy 1.1.1. `SRC_EINSUM_OPTIMIZE` unset (src-method's default `dp` paths). Jobs: 62438530, 62439612, 62442407, 62442408.

## Checks

| check | value |
|---|---|
| apply_vs_dense_L8 | 1.58e-15 |
| apply_exact_vs_dense_L8 | 3.51e-16 |
| compress_sum_vs_dense_L8 | 2.06e-15 |
| inner_vs_dense_L8 | 1.35e-14 |
| quimb-zipup_apply_vs_dense_L8 | 2.58e-15 |
| quimb-zipup_sum_vs_dense_L8 | 3.25e-15 |
| quimb-src_apply_vs_dense_L8 | 1.17e-15 |
| quimb-src_sum_vs_dense_L8 | 2.24e-15 |
| quimb-fit_apply_vs_dense_L8 | 1.27e-15 |
| quimb-fit_sum_vs_dense_L8 | 2.12e-15 |
| cylinder_2x4_D | 14 |
| cylinder_2x4_vs_dense | 0.00e+00 |
| cylinder_4x16_D | 14 |

## Ground states

| model | sites | D | E0 | Emax | a | DMRG sweeps | rel. change | bond | q (k) | t0 bond | mu0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 100 | 5 | -44.1277398929 | 24.750000 | 34.8748 | 7 | 6.2e-11 | 182 | 0.4950 pi (50) | 364 | 0.156522 |
| chain | 100 | 5 | -44.1277398929 | 24.750000 | 34.8748 | 7 | 6.2e-11 | 182 | 0.9901 pi (100) | 364 | 1.483266 |
| chain | 20 | 5 | -8.6824733344 | 4.750000 | 6.8013 | 4 | 3.4e-12 | 51 | 0.5238 pi (11) | 102 | 0.215524 |
| chain | 20 | 5 | -8.6824733344 | 4.750000 | 6.8013 | 4 | 3.4e-12 | 51 | 0.9524 pi (20) | 102 | 0.976802 |
| yc4 | 64 | 14 | -43.0589035377 | 31.000000 | 37.4982 | 9 | 2.2e-11 | 256 | 0.4923 pi (32) | 512 | 0.244335 |
| yc4 | 64 | 14 | -43.0589035377 | 31.000000 | 37.4982 | 9 | 2.2e-11 | 256 | 0.9846 pi (64) | 512 | 2.262852 |

## Phase 1: chain L = 20 against exact moments

DMRG E0 -8.682473334363; <psi0|H|psi0> with the sparse H -8.682473334363. Error `max_n |mu_n - mu_n^exact| / mu_0` over the moments each run reached (all seeds: max).

| q | method | chi | moments | max error | error at last moment |
|---|---|---|---|---|---|
| half | quimb-fit | 64 | 200 | 6.18e-04 | 4.71e-04 |
| half | quimb-fit | 128 | 200 | 3.02e-06 | 1.42e-06 |
| half | quimb-fit | 256 | 200 | 2.62e-09 | 2.27e-10 |
| half | quimb-src | 64 | 200 | 4.03e-02 | 6.57e-03 |
| half | quimb-src | 128 | 200 | 4.45e-04 | 3.37e-04 |
| half | quimb-src | 256 | 200 | 9.14e-07 | 4.17e-07 |
| half | quimb-zipup | 64 | 200 | 7.38e-02 | 1.22e-02 |
| half | quimb-zipup | 128 | 200 | 3.13e-04 | 1.56e-04 |
| half | quimb-zipup | 256 | 200 | 1.23e-06 | 7.72e-08 |
| half | quimb-zipup-rightcanon | 64 | 200 | 1.27e+01 | 7.48e+00 |
| half | quimb-zipup-rightcanon | 128 | 200 | 3.86e+00 | 2.19e+00 |
| half | quimb-zipup-rightcanon | 256 | 200 | 9.65e-01 | 1.36e-01 |
| half | src-method | 64 | 200 | 3.14e-02 | 8.90e-03 |
| half | src-method | 128 | 200 | 4.62e-04 | 3.24e-04 |
| half | src-method | 256 | 200 | 6.96e-07 | 4.05e-07 |
| pi | quimb-fit | 64 | 200 | 7.09e-04 | 7.58e-05 |
| pi | quimb-fit | 128 | 200 | 2.58e-06 | 7.64e-07 |
| pi | quimb-fit | 256 | 200 | 1.45e-09 | 7.13e-10 |
| pi | quimb-src | 64 | 200 | 7.03e-02 | 1.98e-02 |
| pi | quimb-src | 128 | 200 | 5.80e-04 | 6.45e-06 |
| pi | quimb-src | 256 | 200 | 1.02e-06 | 5.21e-07 |
| pi | quimb-zipup | 64 | 200 | 1.13e-01 | 1.05e-02 |
| pi | quimb-zipup | 128 | 200 | 9.90e-04 | 3.70e-05 |
| pi | quimb-zipup | 256 | 200 | 1.79e-06 | 4.93e-07 |
| pi | quimb-zipup-rightcanon | 64 | 200 | 3.52e+00 | 1.06e+00 |
| pi | quimb-zipup-rightcanon | 128 | 200 | 1.48e+00 | 1.38e-01 |
| pi | quimb-zipup-rightcanon | 256 | 200 | 4.71e-01 | 1.11e-01 |
| pi | src-method | 64 | 200 | 5.63e-02 | 9.27e-03 |
| pi | src-method | 128 | 200 | 4.71e-04 | 8.40e-05 |
| pi | src-method | 256 | 200 | 4.91e-07 | 2.74e-07 |

![phase1_L20_moment_error.png](phase1_L20_moment_error.png)

## chain L = 100 (MPO bond D = 5)

### q = 0.4950 pi, 144 moments

Reference: src-method chi = 512, seed 0. Against chi = 256: max moment difference 1.1e-02 (of mu_0), S(q, w) difference 1.0e-04 (relative L2 over w in [0, 4]).

Physics check: edges 1.571 and 2.204; with the kernel width pi a / N = 0.761, 98.8% of the reference weight lies in [0.810, 2.965].

| method | chi | seeds | steps | apply s | compress s | s / moment | time for 144 moments s | pool GB | max moment error | S error |
|---|---|---|---|---|---|---|---|---|---|---|
| src-method | 128 | 3 | 100 | 0.231 | 0.245 | 0.238 | 34 | 0.8 | 2.6e-01 | 7.2e-03 |
| src-method | 256 | 3 | 100 | 0.352 | 0.709 | 0.530 | 76 | 1.6 | 1.2e-02 | 1.1e-04 |
| src-method | 512 | 1 | 72 | 0.915 | 2.551 | 1.733 | 250 | 5.2 | ref | ref |

![chain_L100_half_time.png](chain_L100_half_time.png)

![chain_L100_half_spectrum.png](chain_L100_half_spectrum.png)

### q = 0.9901 pi, 200 moments

Reference: src-method chi = 512, seed 0. Against chi = 256: max moment difference 1.6e-01 (of mu_0), S(q, w) difference 5.1e-04 (relative L2 over w in [0, 4]).

Physics check: edges 0.049 and 3.141; with the kernel width pi a / N = 0.548, 100.0% of the reference weight lies in [0.000, 3.689].

| method | chi | seeds | steps | apply s | compress s | s / moment | time for 200 moments s | pool GB | max moment error | S error |
|---|---|---|---|---|---|---|---|---|---|---|
| src-method | 128 | 3 | 100 | 0.230 | 0.245 | 0.238 | 48 | 0.8 | 7.1e-01 | 8.8e-03 |
| src-method | 256 | 3 | 100 | 0.347 | 0.708 | 0.528 | 106 | 1.6 | 1.6e-01 | 5.1e-04 |
| src-method | 512 | 1 | 100 | 0.915 | 2.593 | 1.754 | 351 | 5.2 | ref | ref |
| quimb-zipup | 128 | 1 | 92 | 2.259 | 2.311 | 2.285 | 457 | 3.8 | - | - |

![chain_L100_pi_time.png](chain_L100_pi_time.png)

![chain_L100_pi_spectrum.png](chain_L100_pi_spectrum.png)
