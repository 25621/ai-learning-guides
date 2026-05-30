# FlashDecoding Ablation

---

> Turn the decode-tuned attention kernel on and off, and measure exactly what it buys.

---

## Key Insight

This project runs the same model in [vLLM](/shared/glossary/#vllm) with and without the [FlashDecoding](/shared/glossary/#flashdecoding) [kernel](/shared/glossary/#kernel) — an [ablation](/shared/glossary/#ablation) — and measures the change in [decode](/shared/glossary/#decode) [throughput](/shared/glossary/#throughput).

## Why This Matters

FlashDecoding reorganizes how the [KV cache](/shared/glossary/#kv-cache) is read so the GPU keeps its [HBM](/shared/glossary/#hbm) bandwidth saturated during decode. Measuring the on/off gap shows how much a single well-matched kernel is really worth.
