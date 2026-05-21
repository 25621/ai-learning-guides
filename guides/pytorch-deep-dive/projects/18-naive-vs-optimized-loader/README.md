# Naive vs Optimized Loader

---

> A fast GPU sitting idle, waiting for the next batch, is the most expensive way to do nothing.

---

## Key Insight

A [DataLoader](/shared/glossary/#dataloader) can prepare upcoming batches using background [worker processes](/shared/glossary/#worker-processes) while the GPU trains on the current one. Raising the number of workers from 0 to several spreads this preparation across CPU cores, so the GPU rarely has to wait for data.

## Why This Matters

A starved GPU is wasted money — your most expensive hardware sitting idle. Tuning the worker count is often the easiest way to raise training [throughput](/shared/glossary/#throughput), sometimes several-fold, without changing the model at all.
