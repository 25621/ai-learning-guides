# KV Cache From Scratch

---

> The fastest way to understand the KV cache is to delete it, watch decode crawl, then add it back.

---

## Key Insight

This project bolts a simple, contiguous [KV cache](/shared/glossary/#kv-cache) onto a small [transformer](/shared/glossary/#transformer) and checks that generating *with* the cache produces exactly the same tokens as generating *without* it — bit-for-bit. The cache stores the [attention](/shared/glossary/#attention) keys and values from earlier tokens, so each [decode](/shared/glossary/#decode) step only has to compute them for the single new token.

## Why This Matters

Without a cache, every new token re-reads and re-computes the entire prompt, so generation gets slower the longer it runs. Building the cache yourself — and proving it changes speed but not output — is the cleanest way to trust that this optimization is safe before you rely on it in a real serving engine.
