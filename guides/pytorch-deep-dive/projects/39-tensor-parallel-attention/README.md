# Tensor Parallel Attention

---

> When one layer is too big for one GPU, cut the layer itself in half.

---

## Key Insight

[Tensor parallelism](/shared/glossary/#tensor-parallelism-tp) splits the [weights](/shared/glossary/#weights) of a single layer across GPUs, instead of replicating the whole model. Splitting a multi-head [attention](/shared/glossary/#attention) layer [column-wise](/shared/glossary/#column-wise-partitioning) across two GPUs (the [Megatron](/shared/glossary/#megatron) style) lets each GPU compute part of the [heads](/shared/glossary/#heads) and then combine the results.

## Why This Matters

Some layers are too large to fit or run on one GPU. Tensor parallelism is the standard way to spread that single layer's work across several GPUs, and it is a core building block for training the very largest models.
