# Stateful Sessions

---

> A session is just a [KV cache](/shared/glossary/#kv-cache) you keep alive between calls — and a plan for what to throw away when memory runs out.

---

## Key Insight

This project builds a session API that keeps each conversation's [KV cache](/shared/glossary/#kv-cache) alive across separate requests, so every new turn reuses the work of every earlier turn instead of re-running [prefill](/shared/glossary/#prefill) over the whole history. The harder half is eviction: GPU memory is finite, so when many sessions are active at once the system must cleanly drop the coldest ones and rebuild them later if they come back.

## Why This Matters

[Multi-turn](/shared/glossary/#multi-turn-conversation) chat and long-running agents grow their history every turn, and re-reading the full transcript each time is both slow and wasteful. Treating the cache as a session-lifetime resource — created, reused, and evicted under pressure — is what lets a [multi-tenant](/shared/glossary/#multi-tenant) system stay fast for warm sessions without falling over when too many arrive at once.
