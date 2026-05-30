# Attention-Sink Eviction

---

> Most past tokens barely matter for the next word — but the first few always do, so never throw those away.

---

## Key Insight

This project implements an H2O-style eviction policy: when the [KV cache](/shared/glossary/#kv-cache) grows too large, it drops the tokens that have been getting little [attention](/shared/glossary/#attention) while always keeping the first few tokens — the [attention sink](/shared/glossary/#attention-sink) — and then measures answer quality at long [context](/shared/glossary/#context-window).

## Why This Matters

For long contexts the cache becomes the memory bottleneck, yet most old tokens contribute almost nothing to the next prediction. Keeping only the important tokens plus the attention sink lets you serve much longer sequences in the same memory with little quality loss.
