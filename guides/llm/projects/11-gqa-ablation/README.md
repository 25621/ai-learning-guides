# GQA Ablation

---

> Fewer key/value heads means a smaller cache — the cheapest way to make a model serve faster.

---

## Key Insight

[Multi-head attention](/shared/glossary/#multi-head-attention) gives every query [head](/shared/glossary/#heads) its own keys and values; [Multi-Query Attention (MQA)](/shared/glossary/#mqa) shares a single key/value head across all of them; [Grouped-Query Attention (GQA)](/shared/glossary/#gqa) sits in between. Fewer key/value heads means a smaller [KV cache](/shared/glossary/#kv-cache), at some cost to quality.

## Why This Matters

The KV cache dominates memory at serving time, so this trade-off decides how many users a model can handle at once. Training matched 100M models with MHA, GQA, and MQA lets you measure the cache savings against the [validation-loss](/shared/glossary/#validation-loss) cost directly.
