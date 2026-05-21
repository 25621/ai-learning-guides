# Profile a Training Step

---

> Every slow training step hides its secret in a few greedy kernels.

---

## Key Insight

The PyTorch [profiler](/shared/glossary/#profiler) records how long every operation — every GPU [kernel](/shared/glossary/#kernel) — takes during one forward, backward, and optimizer step. Because [CUDA](/shared/glossary/#cuda) runs work asynchronously, ordinary timers mislead; the profiler captures true GPU time so you can rank kernels and see which few dominate.

## Why This Matters

Optimization only pays off when aimed at the real hot spot. Ranking kernels by time tells you exactly where to look, so you tune the operations that actually cost you and ignore the ones that don't.
