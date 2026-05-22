# Two-GPU DDP

---

> Two GPUs, two copies of the model, one shared gradient — and almost twice the speed.

---

## Key Insight

[DDP](/shared/glossary/#ddp) puts a full copy of the model on each GPU, splits the batch between them, and averages the [gradients](/shared/glossary/#gradients) so every copy stays in sync. You launch the job with [torchrun](/shared/glossary/#torchrun), which starts one process per GPU and wires them together.

## Why This Matters

DDP is the simplest and most common way to train faster. Watching two GPUs give nearly double the [throughput](/shared/glossary/#throughput) builds the intuition you need before moving on to sharded or multi-node training.
