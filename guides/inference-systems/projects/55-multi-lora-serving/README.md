# Multi-LoRA Serving

---

> One base model in memory, a thousand fine-tunes on top — no replica explosion.

---

## Key Insight

This project stands up a multi-adapter engine (Lorax or S-LoRA), trains 5 small [LoRA](/shared/glossary/#lora) adapters, and serves them all from a *single* copy of the base model, then compares [throughput](/shared/glossary/#throughput) against running 5 separate replicas. The trick is batching requests that use *different* adapters into one forward pass — see [multi-LoRA](/shared/glossary/#multi-lora).

## Why This Matters

Giving every customer their own fine-tuned model would normally mean one GPU deployment per customer, which does not scale. Because each adapter is tiny (megabytes) while the base model is large, sharing one base across hundreds of adapters is the economic feature that makes per-tenant fine-tuning affordable for SaaS products.
