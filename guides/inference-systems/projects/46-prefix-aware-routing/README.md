# Prefix-Aware Routing

---

> Send requests that share an opening to the same replica, so its cache is already warm.

---

## Key Insight

This project builds a small routing layer that sends requests with the same starting tokens to the same replica — so that replica's [prefix cache](/shared/glossary/#prefix-cache) already holds their shared [system prompt](/shared/glossary/#system-prompt) — and verifies the cache hit rate goes up.

## Why This Matters

When many users share a long system prompt, routing them to the same replica turns a fresh, slow [prefill](/shared/glossary/#prefill) into a near-instant cache hit, sharply cutting [time to first token](/shared/glossary/#ttft) in multi-tenant systems.
