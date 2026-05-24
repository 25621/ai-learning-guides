# Tokenizer Compression Study

---

> The same sentence costs many more tokens in some languages than others — and you pay per token.

---

## Key Insight

A [tokenizer](/shared/glossary/#tokenizer) trained mostly on English splits other languages into far more pieces, so identical meaning takes more tokens. Measuring [tokens per byte](/shared/glossary/#tokens-per-byte) across languages exposes this hidden imbalance.

## Why This Matters

More tokens means higher cost, slower replies, and a smaller effective [context window](/shared/glossary/#context-window) — which is why non-English users often get a worse, pricier experience from the very same model.
