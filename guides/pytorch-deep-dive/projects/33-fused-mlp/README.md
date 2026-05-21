# Fused MLP

---

> Four operations, one trip to memory — fusion turns a pipeline into a single kernel.

---

## Key Insight

An [MLP](/shared/glossary/#mlp) normally runs as separate steps — [matmul](/shared/glossary/#matmul), add bias, [GELU](/shared/glossary/#gelu), matmul — each reading and writing memory. [Kernel fusion](/shared/glossary/#kernel-fusion) merges them into one [Triton](/shared/glossary/#triton) [kernel](/shared/glossary/#kernel) that keeps the intermediate results on-chip, so the data is read once instead of four times.

## Why This Matters

Most deep-learning operations are limited by memory bandwidth, not arithmetic, so fusing several small ops into one is among the most reliable ways to make a model faster.
