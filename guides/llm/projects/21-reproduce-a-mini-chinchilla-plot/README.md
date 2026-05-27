# Reproduce a Mini-Chinchilla Plot

---

> Train seven small models and watch the scaling law draw its own curve.

---

## Key Insight

[Scaling laws](/shared/glossary/#scaling-laws) say a model's [loss](/shared/glossary/#loss-function) falls in a smooth, predictable way as you add [parameters](/shared/glossary/#parameters), data, and compute. Training seven models from 10M to 500M parameters — each given the right number of tokens for its size — and plotting their [iso](/shared/glossary/#iso)-[FLOP](/shared/glossary/#flops) loss curves reproduces the [Chinchilla](/shared/glossary/#chinchilla) result in miniature: for a fixed compute budget, there is one model size that wins.

## Why This Matters

Seeing the curve emerge from your own runs makes scaling laws believable. They are what let a lab predict a giant model's loss from a handful of small ones — the forecast that justifies betting millions of dollars on a single training run.
