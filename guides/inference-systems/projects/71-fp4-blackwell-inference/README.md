# FP4 (Blackwell) Inference

---

> [FP4](/shared/glossary/#fp4) gives you four bits per weight and 16 possible values — astonishing memory savings, and a quality cliff you have to measure, not assume.

---

## Key Insight

This project benchmarks [FP4](/shared/glossary/#fp4) weights and [activations](/shared/glossary/#activations) against the [FP8](/shared/glossary/#fp8) baseline on [Blackwell](/shared/glossary/#blackwell) hardware, which accelerates 4-bit math natively, and then checks quality on a real eval. The point is to see both sides of the trade at once: how much memory and speed you gain, and how much accuracy (if any) you give up by halving the bits again.

## Why This Matters

[Quantization](/shared/glossary/#quantization) is the most leveraged cost knob in serving, and FP4 pushes it to the edge of what is usable — fitting models on fewer chips than ever before. But "lossless 4-bit" is a claim to verify, never trust: with only 16 representable values, quality can drop in ways that only a workload-matched eval will catch before customers do.
