# Prediction: strong scaling on 8x RTX PRO 6000 over PCIe, complex64

Committed before any multi-GPU run of step 4. Inputs: the 1-GPU traced run at the
reference size (`raw/trace_g1_m4000.json`, `step3_reference.sh`) and the bandwidth
probes (`raw/probe.json`, `env.json`). Produced by `predict.py`
(`raw/prediction.json`).

Reference size: 50 sites, `D_M = 4000`, `chi_out = 2000`, complex64 (`e = 8` B),
stack in `/dev/shm`, default budgets: 7 environments on the device, 43 in host
memory, none on disk.

## Single GPU (measured)

| Quantity | Value |
|---|---|
| Wall time `T_1` (traced run) | 843.4 s |
| Left-to-right / right-to-left pass | 236.3 / 603.7 s |
| Right-to-left per bulk site (GPU span) | 12.8 s (12.2 s averaged over 49 sites) |
| Right-to-left flops along the executed paths | 3.68e16 |
| Right-to-left rate, kernels only / whole GPU span | 63.5 / 61.7 TFLOP/s |
| Left-to-right rate, kernels only | 57.5 TFLOP/s |
| QR on rank 0, per site / all sites | 56 ms / 2.8 s |
| complex64 GEMM peak (8192^3, `env.json`) | 84 TFLOP/s |

The right-to-left pass runs at 75% of the measured complex64 GEMM rate.

## Model

`T_G = (T_1 - S_1) / G + t_QR + t_comm(G) + t_upload(G)`, with
`S_1 = t_QR + t_upload(1)` the part of `T_1` that does not divide (3.6 s of core
uploads at 56.4 GB/s plus 2.8 s of QR).

Per site, complex64 payloads (the paper's Table "comm", halved):

| Collective | Bytes | Bandwidth used (measured, concurrent) |
|---|---|---|
| gather `M_j` to rank 0 | `(G-1)/G x 0.512 GB` | gather into GPU 0: 41.2 / 41.1 / 39.1 GB/s at G = 2/4/8 |
| scatter `Gamma_j` | `0.512 GB / G` per receiver | GPU 0 to all: 41.4 / 13.7 / 5.6 GB/s per receiver |
| allgather `S_j` | `(G-1)/G x 4.096 GB` per receiver | all-to-all: 37.2 / 21.3 / 15.7 GB/s per receiver |
| cores, each pass (`t_upload`) | `2.05 GB / G` H2D, then `(G-1)/G x 2.05 GB` allgather | H2D 56.5 / 46.7 / 32.4 GB/s per GPU; allgather as above |

## Prediction

| G | gather | scatter | allgather `S` | cores (per pass) | QR | `t_comm` | `t_upload` | `T_G` (s) | Speed-up | Efficiency |
|---|---|---|---|---|---|---|---|---|---|---|
| | ms/site | ms/site | ms/site | ms/site | ms/site | s | s | | | |
| 1 | | | | 36 | 56 | | 3.6 | 843.4 | 1.00 | 100% |
| 2 | 6 | 6 | 55 | 46 | 56 | 3.3 | 4.5 | 429 | 1.97 | **98%** |
| 4 | 9 | 9 | 144 | 83 | 56 | 8.0 | 8.1 | 228 | 3.70 | **92%** |
| 8 | 12 | 12 | 229 | 122 | 56 | 12.3 | 12.0 | 132 | 6.40 | **80%** |

At `G = 8` the serial and communication terms add up to 27 s against 105 s of
divided computation. The allgather of `S` dominates the communication; its
per-receiver rate falls from 37 to 16 GB/s because all eight GPUs pull at once
over PCIe and half of the pairs cross the UPI link between the sockets
(`nvidia-smi topo`: GPUs 0-3 on socket 0, 4-7 on socket 1, every pair `NODE` or
`SYS`, no PCIe switch shared between GPUs).

## What the model leaves out

- **Changes of plan with G.** Each GPU holds `1/G` of every environment: 2.05 GB
  at `G = 2`, 0.51 GB at `G = 8`, so more environments stay on the device as `G`
  grows (at `G = 8` all 49 should fit). That removes host-to-device environment
  reads and can make the measured efficiency exceed the model, as the 2-GPU
  Leonardo point did. The sketch and projection batches also change with `G`,
  and with them the GEMM efficiency.
- **Load imbalance and synchronization.** Every right-to-left site is
  bulk-synchronous; the model charges the collectives at their bandwidth but not
  the waits for the slowest rank, nor the host-side barriers.
- **Contention between transfers.** Environment uploads, core uploads and peer
  copies share each GPU's PCIe link; the probes measured each pattern alone.
- **NUMA placement.** The page-locked staging buffers are allocated by
  whichever thread first touches them; uploads to GPUs of the other socket cross
  UPI. The concurrent H2D probe (32.4 GB/s per GPU at `G = 8`) includes that.
- **First-run effects.** `scaling` runs the counts in one process, so the
  1-GPU run also pays the one-off costs (pool growth, library handles) that the
  later counts reuse.
