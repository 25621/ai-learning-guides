# Cache-Aware Admission

---

> Don't let a request in the door if there is no room left for its cache.

---

## Key Insight

This project implements [admission control](/shared/glossary/#admission-control) that estimates how much [KV cache](/shared/glossary/#kv-cache) a new request will need and refuses it when the GPU cannot fit it — then verifies the server never runs out of memory.

## Why This Matters

If a [scheduler](/shared/glossary/#scheduler) admits more requests than the cache can hold, the whole server can crash with an out-of-memory error and drop everyone's work at once. Checking the projected cache size *before* admitting keeps the system stable even when traffic spikes past what it can handle.
