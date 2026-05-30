# FP8 KV Cache

---

> Store the cache in 8 bits instead of 16 — almost free speed, almost no quality cost.

---

## Key Insight

This project switches a deployment's [KV cache](/shared/glossary/#kv-cache) from 16-bit ([bfloat16](/shared/glossary/#bfloat16)) to [FP8](/shared/glossary/#fp8), measures the [decode](/shared/glossary/#decode) speedup, and then confirms with a [quality gate](/shared/glossary/#quality-gate) that the answers did not change.

## Why This Matters

Decode speed is set by how many bytes the cache streams from [HBM](/shared/glossary/#hbm) each step, so halving the cache nearly halves that traffic. Because keys and values tolerate low precision well, FP8 KV is one of the safest big wins in serving.
