# Torch Compile Test

---

> Write the model once in Python; let the compiler rewrite it into fast kernels.

---

## Key Insight

[`torch.compile`](/shared/glossary/#torchcompile) traces a model into a graph and generates optimized, [fused kernels](/shared/glossary/#kernel-fusion), turning many small operations into a few large ones. Compiling a [transformer](/shared/glossary/#transformer) block and timing it against [eager mode](/shared/glossary/#eager-mode) shows the speedup from cutting Python overhead and [kernel](/shared/glossary/#kernel) launches.

## Why This Matters

A single line — `torch.compile(model)` — can speed up training with no change to the model's math, which makes it one of the cheapest wins in modern PyTorch whenever the graph compiles cleanly.
