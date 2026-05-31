# Prefix KV Caching

---

> Compute a document's attention once, reuse it forever.

---

## Key Insight

This project pre-computes and stores the [KV cache](/shared/glossary/#kv-cache) for 1000 retrieved documents ahead of time, then measures the [time to first token](/shared/glossary/#ttft) when a query hits a *cold* document (cache must be built) versus a *warm* one (cache already exists). It is the serving-side core of a [RAG](/shared/glossary/#rag) system — see [prefix cache](/shared/glossary/#prefix-cache).

## Why This Matters

In retrieval-augmented serving the same documents are read over and over by different users, and re-running [prefill](/shared/glossary/#prefill) on each one is pure wasted work. Caching the document's keys and values turns a slow first token into a near-instant one, which is often the single biggest latency win available to a RAG product.
