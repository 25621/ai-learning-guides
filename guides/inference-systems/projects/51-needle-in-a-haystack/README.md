# Needle-in-a-Haystack

---

> A model with a 1M-token window is only useful up to the length where it can still find the needle.

---

## Key Insight

This project hides a single fact (the "needle") at different positions inside an ever-longer prompt (the "haystack") and asks the model to recall it, pushing the [context window](/shared/glossary/#context-window) up toward your engine's limit. Plotting recall against length reveals a *cliff* — a point where accuracy suddenly drops even though the engine still accepts the input. See [needle-in-a-haystack](/shared/glossary/#needle-in-a-haystack).

## Why This Matters

A serving engine will happily accept a 200k-token prompt and build a giant [KV cache](/shared/glossary/#kv-cache) for it — but if the model stops actually *using* the far-away tokens, you are paying for memory and compute that buy you nothing. Knowing the real usable length lets you set honest limits instead of advertising a number the model cannot deliver.
