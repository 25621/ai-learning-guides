# Custom Collate

---

> A batch is not just a pile of samples — something has to stack them, and you get to decide how.

---

## Key Insight

A [collate function](/shared/glossary/#collate-function) is the step that combines a list of individual samples into one batched tensor. The default one assumes every sample is the same size; a custom collate function lets you handle variable-length data by [padding](/shared/glossary/#padding) each sample up to the longest one in the batch.

## Why This Matters

Text, audio, and other sequence data rarely come in equal lengths. A custom collate function is what makes batching such data possible at all, and padding per-batch (instead of to one global maximum) avoids a lot of wasted computation.
