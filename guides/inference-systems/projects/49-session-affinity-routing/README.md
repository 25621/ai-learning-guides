# Session-Affinity Routing

---

> Keep a multi-turn chat on the same replica so its cache survives between turns.

---

## Key Insight

This project implements "sticky" routing that sends every turn of a conversation (keyed by its `session_id`) to the same replica, so that replica's [KV cache](/shared/glossary/#kv-cache) from earlier turns is still there — then verifies the multi-turn cache hit rate.

## Why This Matters

Without stickiness, each chat turn might land on a different replica with a cold cache, forcing an expensive re-[prefill](/shared/glossary/#prefill) of the whole history. Affinity routing preserves the cache across turns and keeps long conversations fast.
