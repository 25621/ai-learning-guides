# Memory Breakdown

---

> Training memory is four buckets — know which one overflows.

---

## Key Insight

Training memory has four parts: the model parameters, their [gradients](/shared/glossary/#gradients), the [optimizer state](/shared/glossary/#optimizer-state) ([Adam](/shared/glossary/#adam) keeps two extra values per parameter), and the [activations](/shared/glossary/#activations) saved for the backward pass. Summing these predicts usage, which `torch.cuda.memory_summary()` then confirms.

## Why This Matters

When you hit out-of-memory, knowing which bucket is largest tells you which lever to pull — a smaller batch, [gradient checkpointing](/shared/glossary/#gradient-checkpointing), or a lighter optimizer — instead of guessing.
