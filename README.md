# Fused and Online Softmax Kernels in Triton

Two row-wise softmax GPU kernels written in [Triton](https://github.com/triton-lang/triton), verified against `torch.softmax` and benchmarked against PyTorch.

| Kernel | Idea | When to use |
|---|---|---|
| **Fused softmax** | Load a whole row into on-chip memory, compute max / exp / sum / divide there, write once. One global-memory round-trip instead of ~5 for a naive implementation. | Rows that fit on-chip (up to a few thousand columns) |
| **Online (tiled) softmax** | Stream the row in fixed-size tiles, keeping a running max and normalizer and rescaling the partial sum whenever the max grows (the same trick FlashAttention uses). | Arbitrarily wide rows, e.g. 50k–130k vocabulary logits, with constant on-chip memory |

Both kernels subtract the row maximum before exponentiating, so they are numerically stable for very large or very negative inputs.

## Run

Open [`fused_softmax_triton.ipynb`](fused_softmax_triton.ipynb) in Google Colab, set **Runtime → Change runtime type → T4 GPU**, and run all cells. The notebook:

1. checks correctness on 7 shapes (1×1 up to 16×131,072, including non-power-of-two widths) plus extreme-value inputs
2. benchmarks effective bandwidth (GB/s) for the fused kernel vs `torch.softmax` vs a naive unfused version
3. benchmarks wide rows: online vs fused vs `torch.softmax`
4. profiles kernel launches with the PyTorch profiler

## Results

Measured on a **Tesla T4** (Google Colab; PyTorch 2.11, Triton 3.6). GB/s is effective bandwidth (one read + one write of the matrix); the T4's peak is about 320 GB/s.

**Correctness:** both kernels match `torch.softmax` on all 7 shapes (max absolute error ≤ 4.2e-7) and on extreme inputs (±1000, all −10,000).

**Fused kernel, 4096 rows:**

| Columns | Triton fused | `torch.softmax` | Naive (unfused) | vs naive | vs torch |
|---:|---:|---:|---:|---:|---:|
| 256 | 153.6 | 205.2 | 52.1 | 2.95× | 0.75× |
| 512 | 222.5 | 216.9 | 55.4 | 4.02× | 1.03× |
| 1024 | 227.6 | 212.4 | 57.7 | 3.95× | 1.07× |
| 2048 | 229.8 | 211.6 | 59.2 | 3.89× | 1.09× |
| 4096 | 231.9 | 241.1 | 60.0 | 3.87× | 0.96× |
| 8192 | 229.6 | 234.6 | 60.9 | 3.77× | 0.98× |

The fused kernel is about **3.9× faster than the naive version**, runs at ~72% of the T4's peak bandwidth, and is on par with PyTorch's hand-tuned kernel (0.96–1.09×) from 512 columns up. At 256 columns each row is too small to keep the GPU busy with one row per program.

**Wide rows, 256 rows (online vs fused vs PyTorch):**

| Columns | Online (tiled) | Fused | `torch.softmax` | online vs fused | online vs torch |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 151.2 | 226.1 | 150.7 | 0.67× | 1.00× |
| 32,768 | 156.3 | 224.5 | 126.6 | 0.70× | 1.23× |
| 65,536 | 157.5 | 155.4 | 124.8 | 1.01× | 1.26× |
| 131,072 | 156.3 | 53.0 | 124.5 | **2.95×** | **1.26×** |

Once a row no longer fits on-chip, the fused kernel's throughput collapses (226 → 53 GB/s) because the whole row is held in registers and spills. The online kernel keeps a constant tile in registers and holds steady at ~156 GB/s at every width, which is **2.95× faster than the fused kernel and 1.26× faster than `torch.softmax`** at 131,072 columns.

**Profiling (4096 × 4096, 10 calls, `torch.profiler`):** the naive version launches **5 kernels per call** (50 launches, 23.9 ms of GPU time); the fused version launches **1** (10 launches, 5.5 ms), a 4.3× reduction in GPU time.

## What I'd try next

* fp16 / bf16 inputs (halves memory traffic)
* processing several short rows per program to fix the 256-column case
* a single-pass online kernel that keeps the tile in shared memory when it fits, to close the 16k–32k gap with the fused kernel
