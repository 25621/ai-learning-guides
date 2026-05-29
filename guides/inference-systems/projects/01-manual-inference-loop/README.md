# Manual Inference Loop

---

> Every inference engine is, at heart, a `while` loop that grows a [KV cache](/shared/glossary/#kv-cache) one token at a time.

---

## Key Insight

This project loads a 1B-parameter [LLM](/shared/glossary/#llm), runs the whole prompt through it once to build the [KV cache](/shared/glossary/#kv-cache) ([prefill](/shared/glossary/#prefill)), and then walks a `while` loop that asks the model for one new token at a time ([decode](/shared/glossary/#decode)), feeding the cache back in each step. Writing the two phases by hand makes their very different shapes — one big parallel pass, then many tiny serial steps — concrete.

## Why This Matters

The two-phase split is the single most important fact in LLM serving: [prefill](/shared/glossary/#prefill) is compute-heavy, [decode](/shared/glossary/#decode) is memory-bandwidth-heavy, and almost every later optimization in this guide targets one phase or the other. Building the loop yourself first means you can read any production engine's source code and recognize what it is doing.
