# Disaggregated Prototype

---

> One pool reads the prompt, another writes the answer, and the cache travels between them.

---

## Key Insight

This project builds a prototype of [disaggregated serving](/shared/glossary/#disaggregated-serving) where one process runs [prefill](/shared/glossary/#prefill) and hands its [KV cache](/shared/glossary/#kv-cache) blocks to a separate [decode](/shared/glossary/#decode) process — over shared memory or [RDMA](/shared/glossary/#rdma) — and measures the transfer overhead against doing both in one process.

## Why This Matters

Prefill is compute-heavy while decode is bandwidth-heavy, so giving each its own pool of GPUs lets you size hardware for each job independently. The prototype shows whether moving the cache between pools costs less than that flexibility is worth.
