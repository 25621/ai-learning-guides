# Padding Waste Audit

---

> Every padding token is compute the GPU spends on a word that isn't even there.

---

## Key Insight

This project instruments a static-batching server to measure what fraction of its [decode](/shared/glossary/#decode) [FLOPs](/shared/glossary/#flops) are spent on [padding](/shared/glossary/#padding) — the filler tokens added so every sequence in a [batch](/shared/glossary/#batch) reaches the same length.

## Why This Matters

Padding is pure waste: the GPU does real work on tokens that carry no information. Putting a number on that waste shows exactly why [continuous batching](/shared/glossary/#continuous-batching), which needs no padding, beats static batching on busy, mixed-length traffic.
