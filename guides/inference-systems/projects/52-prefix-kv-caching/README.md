# Prefix KV Caching

---

> Compute a document's attention once, reuse it forever.

---

## Key Insight

This project pre-computes a [KV cache](/shared/glossary/#kv-cache) for each of 1000 retrieved documents — *one separate cache per document*, not a single combined cache for all 1000 — and stores them all ahead of time, then measures the [time to first token](/shared/glossary/#ttft) when a query hits a *cold* document (its cache must still be built) versus a *warm* one (its cache already exists). Because each document is cached on its own, any document can be reused independently, in any combination, by later queries. It is the serving-side core of a [RAG](/shared/glossary/#rag) system — see [prefix cache](/shared/glossary/#prefix-cache).

## Why This Matters

In retrieval-augmented serving the same documents are read over and over by different users, and re-running [prefill](/shared/glossary/#prefill) on each one is pure wasted work. Caching the document's keys and values turns a slow first token into a near-instant one, which is often the single biggest latency win available to a RAG product.
