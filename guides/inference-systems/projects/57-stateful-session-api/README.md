# Stateful Session API

---

> A multi-turn conversation is just a KV cache you refuse to throw away.

---

## Key Insight

This project builds a session API that keeps a conversation's [KV cache](/shared/glossary/#kv-cache) alive *across* separate calls, so each new turn reuses the work of every previous turn instead of re-reading the whole history. Because GPU memory is finite, it must also evict idle sessions under pressure — and the project verifies the cache-hit rate to confirm warm sessions are actually being reused.

## Why This Matters

In long agent and chat sessions the history grows every turn, and re-running [prefill](/shared/glossary/#prefill) over the full transcript each time is both slow and expensive. Treating the cache as a session-lifetime resource — created, reused, and evicted on demand — is what makes multi-turn, [multi-tenant](/shared/glossary/#multi-tenant) systems both fast and stable under load.
