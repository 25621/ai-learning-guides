# KV-Quantization Study

---

> Storing the cache in 8 bits instead of 16 nearly doubles how many users fit — usually with no quality drop you can measure.

---

## Key Insight

This project stores the [KV cache](/shared/glossary/#kv-cache) in a smaller number format — [FP8](/shared/glossary/#fp8) or [int8](/shared/glossary/#int8) instead of 16-bit — and measures two things: whether answer quality drops on a held-out test set, and how much [throughput](/shared/glossary/#throughput) goes up. This is [quantization](/shared/glossary/#quantization) applied to the cache rather than to the weights.

## Why This Matters

The cache is large, and [decode](/shared/glossary/#decode) speed is set by how many bytes it must read each step, so halving its size nearly halves that traffic. Because keys and values tolerate low precision well, this is one of the cheapest wins in serving — but only a real evaluation proves the quality actually held.
