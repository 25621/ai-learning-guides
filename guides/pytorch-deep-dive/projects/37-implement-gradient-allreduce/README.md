# Implement Gradient AllReduce

---

> DDP's magic is one collective call — write it yourself and the magic disappears.

---

## Key Insight

[AllReduce](/shared/glossary/#allreduce) is the [collective operation](/shared/glossary/#collective-operation) that sums a tensor across every [rank](/shared/glossary/#rank) and hands the result back to all of them. Doing this by hand on your [gradients](/shared/glossary/#gradients) with `torch.distributed.all_reduce` reproduces exactly what [DDP](/shared/glossary/#ddp) does automatically.

## Why This Matters

Once you can write the all-reduce yourself, DDP stops being a black box. You will understand why every GPU ends up with identical gradients, and therefore the same model, after each step.
