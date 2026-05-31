# Tiny Paged Cache

---

> Stop giving each request one big contiguous slab; hand out small fixed-size pages instead, and the wasted space disappears.

---

## Key Insight

This project builds a small [PagedAttention](/shared/glossary/#pagedattention)-style [KV cache](/shared/glossary/#kv-cache): instead of one contiguous block per request, the cache is cut into fixed-size pages (16 tokens each) that are handed out on demand and tracked by a per-request page table — the same data structure [vLLM](/shared/glossary/#vllm) uses.

## Why This Matters

Giving each request its own contiguous chunk wastes memory through [fragmentation](/shared/glossary/#fragmentation) — gaps too small to reuse. Paging removes those gaps, so the same GPU fits far more concurrent requests; reproducing the page table by hand demystifies the core idea behind every modern serving engine.
