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

Results to be added after the benchmark run.
