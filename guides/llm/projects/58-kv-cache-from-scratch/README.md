# KV Cache from Scratch

---

> The cheapest speedup is the work you don't repeat.

---

## Key Insight

This project adds a [KV cache](/shared/glossary/#kv-cache) to a [transformer](/shared/glossary/#transformer) decoder so each newly generated token reuses the keys and values already computed for earlier tokens instead of recomputing [attention](/shared/glossary/#attention) over the entire prefix, and measures the speedup against a naive recompute-every-step baseline.

## Why This Matters

Without the cache, generating each new token re-reads the whole prompt — so cost grows with sequence length squared and inference crawls at long contexts. With it, decoding scales roughly linearly with sequence length, which is why the KV cache is the foundation every production serving engine is built on.
