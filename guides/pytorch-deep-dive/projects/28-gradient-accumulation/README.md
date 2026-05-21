# Gradient Accumulation

---

> Add up the gradients of several small batches, then step as if the batch were huge.

---

## Key Insight

[Gradient accumulation](/shared/glossary/#gradient-accumulation) runs several small batches, adds up their [gradients](/shared/glossary/#gradients), and only calls the [optimizer](/shared/glossary/#optimizer)'s step after a set number of them. Because gradients add, the result matches one large batch — while only one small batch's [activations](/shared/glossary/#activations) ever sit in memory at once.

## Why This Matters

It lets a small GPU train at a large effective batch size, reproducing results that would otherwise need bigger or more numerous GPUs.
