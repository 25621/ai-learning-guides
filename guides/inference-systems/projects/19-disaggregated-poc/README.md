# Disaggregated PoC

---

> Let one machine read the prompt and another write the answer, and ship the cache between them.

---

## Key Insight

This project builds a small [proof-of-concept](/shared/glossary/#poc) of [disaggregated serving](/shared/glossary/#disaggregated-serving): one process runs [prefill](/shared/glossary/#prefill), a second runs [decode](/shared/glossary/#decode), and they hand off the [KV cache](/shared/glossary/#kv-cache) between them. It then measures the transfer overhead against doing both phases in one process.

## Why This Matters

Prefill is compute-heavy while decode is memory-bandwidth-heavy, so giving each phase its own pool of GPUs lets you size hardware for each job independently. The proof-of-concept shows whether the cost of moving the cache between them is small enough to make that split worthwhile.
