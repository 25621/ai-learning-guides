# Prefix-Aware Routing

---

> Send every request that begins the same way to the same server, so its shared opening is processed once, not over and over.

---

## Key Insight

This project builds a small routing layer that sends requests sharing the same opening tokens to the same replica. Because those requests all start with the same long [system prompt](/shared/glossary/#system-prompt), the first one to arrive makes that replica compute it once and save the result in its [prefix cache](/shared/glossary/#prefix-cache); every later request sent there reuses that saved work instead of redoing it, and the project checks that this raises the cache hit rate. It's like sending everyone who orders the same set menu to the one chef who has prepped the shared starter, instead of scattering them across kitchens that each cook it from scratch.

## Why This Matters

When many users share a long system prompt, routing them to the same replica turns a fresh, slow [prefill](/shared/glossary/#prefill) into a near-instant cache hit, sharply cutting [time to first token](/shared/glossary/#ttft) in [multi-tenant](/shared/glossary/#multi-tenant) systems.

We route the *request* to where the cache already lives, rather than copying the cache to whichever replica the request happens to land on, because that cache is the [KV cache](/shared/glossary/#kv-cache) — often several gigabytes sitting in one replica's GPU memory. Shipping a tiny request over to the right replica is far cheaper than moving gigabytes of cache around the cluster for every call.
