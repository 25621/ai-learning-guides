# Quality-Filter Ablation

---

> Let a classifier throw away the junk, then ask: did keeping only the "good" text actually help?

---

## Key Insight

This [ablation](/shared/glossary/#ablation) repeats the dedup experiment with a different lever — a [quality filter](/shared/glossary/#quality-filter), a classifier that keeps only educational-looking text. Two identical models, one trained on filtered data and one without, reveal the filter's true effect.

## Why This Matters

Aggressive quality filtering (as in [FineWeb-Edu](/shared/glossary/#fineweb-edu)) is one of the biggest drivers of modern model quality, but "quality" is a judgment call baked into a classifier. Running the ablation teaches you to trust measured downstream gains over intuition about what "good data" means.
