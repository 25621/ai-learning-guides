# FSDP a Transformer

---

> Don't copy the whole model to every GPU — give each one a slice and borrow the rest just in time.

---

## Key Insight

[FSDP](/shared/glossary/#fsdp) splits a model's parameters, [gradients](/shared/glossary/#gradients), and [optimizer state](/shared/glossary/#optimizer-state) into shards, one per GPU, and gathers each full layer only for the moment it is needed. This lets you train a [transformer](/shared/glossary/#transformer) that is far too large to fit on a single GPU under [DDP](/shared/glossary/#ddp).

## Why This Matters

FSDP is the modern default for training large models on ordinary clusters. Seeing a model run under FSDP that crashes under DDP makes the memory savings concrete.
