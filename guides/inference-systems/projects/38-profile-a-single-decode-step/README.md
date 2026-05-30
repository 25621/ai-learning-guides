# Profile a Single Decode Step

---

> Open one decode step under a profiler and watch where the microseconds actually go.

---

## Key Insight

This project uses a [profiler](/shared/glossary/#profiler) to record a single [decode](/shared/glossary/#decode) step and pick apart where the time goes — the longest [kernel](/shared/glossary/#kernel), the [HBM](/shared/glossary/#hbm) read pattern, and the overhead of launching many tiny kernels.

## Why This Matters

Decode is memory-bound, so its bottleneck is rarely "the math is slow"; only a profiler trace shows the real culprit. Reading one is the skill that lets you fix a latency regression instead of guessing at it.
