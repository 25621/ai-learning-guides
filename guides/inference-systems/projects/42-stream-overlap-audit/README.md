# Stream-Overlap Audit

---

> While the GPU waits on the CPU, work that could run in parallel is sitting idle.

---

## Key Insight

This project finds a place where work runs in series and stalls the GPU — for example [detokenization](/shared/glossary/#detokenization) on the CPU — then overlaps it with the next GPU forward pass using separate [CUDA streams](/shared/glossary/#cuda-stream) and measures the gain.

## Why This Matters

A serving stack that pipelines CPU and GPU work instead of doing them one after another is often 20–40% faster for free. Auditing for these serial gaps is one of the highest-return tuning steps you can take.
