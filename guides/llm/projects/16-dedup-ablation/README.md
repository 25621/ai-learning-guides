# Dedup Ablation

---

> Same model, same compute — the only difference is whether you removed the duplicates first.

---

## Key Insight

An [ablation](/shared/glossary/#ablation) trains two identical 100M models that differ in one thing only: one sees raw web data, the other sees the same data after [deduplication](/shared/glossary/#deduplication) (here [MinHash](/shared/glossary/#minhash) near-duplicate removal). Comparing their downstream scores isolates exactly what dedup buys you.

## Why This Matters

Removing duplicate documents is one of the highest-return moves in [pretraining](/shared/glossary/#pretraining) — duplicates waste compute and let the model memorize instead of generalize. Measuring the gain yourself shows why data cleaning, not architecture, dominates modern training.
