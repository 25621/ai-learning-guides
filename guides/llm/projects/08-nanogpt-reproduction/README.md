# nanoGPT Reproduction

---

> The smallest honest GPT is a few hundred lines — small enough to hold in your head, real enough to write Shakespeare.

---

## Key Insight

nanoGPT is a minimal but complete decoder-only GPT: token and position [embeddings](/shared/glossary/#embedding), a stack of [transformer](/shared/glossary/#transformer) blocks, and a final linear head that predicts the next token. Typing it out yourself, training on a tiny Shakespeare file, and sampling text exercises the whole loop end to end.

## Why This Matters

Most "magic" disappears once you have trained a real GPT from scratch. nanoGPT is the cleanest reference for that experience — every later optimization (GQA, RoPE, MoE) is a small change to this same skeleton.
