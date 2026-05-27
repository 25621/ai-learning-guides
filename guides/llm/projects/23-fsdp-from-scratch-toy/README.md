# FSDP from Scratch (Toy)

---

> Split the model across two GPUs by hand, and the magic of sharded training disappears.

---

## Key Insight

[FSDP](/shared/glossary/#fsdp) (Fully Sharded Data Parallel) splits a model's [weights](/shared/glossary/#weights), [gradients](/shared/glossary/#gradients), and [optimizer state](/shared/glossary/#optimizer-state) across GPUs so no single GPU has to hold the whole thing. Building a toy version by hand on 2 GPUs — and checking it reaches the same result as plain [data parallelism](/shared/glossary/#data-parallelism) — shows exactly what the library does for you.

## Why This Matters

[AdamW](/shared/glossary/#adamw)'s [optimizer state](/shared/glossary/#optimizer-state) alone is several times the size of the model, so large models do not fit on one GPU. Sharding with FSDP (and its cousin [ZeRO](/shared/glossary/#zero)) is what makes training beyond a few billion parameters possible at all.
