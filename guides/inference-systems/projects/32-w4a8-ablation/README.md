# W4A8 Ablation

---

> 4-bit weights save memory; adding 8-bit activations saves compute — measure what each one really buys.

---

## Key Insight

This project runs an [ablation](/shared/glossary/#ablation) comparing weight-only 4-bit [quantization](/shared/glossary/#quantization) against W4A8 — 4-bit [weights](/shared/glossary/#weights) *plus* 8-bit [activations](/shared/glossary/#activations) — measuring both answer quality and [throughput](/shared/glossary/#throughput) (tokens per second).

## Why This Matters

Quantizing weights mainly saves memory and bandwidth, while quantizing the activations too can speed up the actual math — but activations are harder to compress safely. Measuring the real quality-versus-speed trade-off tells you whether the extra step is worth it for your workload.
