# Weighted Sampler

---

> If 99% of your data is one class, random batches teach the model just one trick: always guess that class.

---

## Key Insight

A [sampler](/shared/glossary/#sampler) decides the order in which a [DataLoader](/shared/glossary/#dataloader) visits examples. A `WeightedRandomSampler` gives each example its own sampling probability, so you can draw rare classes more often and build balanced batches from an imbalanced dataset.

## Why This Matters

On imbalanced data, a model can reach high accuracy by always predicting the majority class while learning nothing useful. Balanced sampling forces the model to see minority classes often enough to actually learn them.
