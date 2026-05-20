# Gradient Checkpointing

---

> Trade compute for memory to train bigger models.

---

## Key Insight

Normally, PyTorch saves all intermediate [activations](/shared/glossary/#activations) during the forward pass to use them in the [backward pass](/shared/glossary/#backward-pass). [Gradient checkpointing](/shared/glossary/#gradient-checkpointing) discards most of these activations to save memory, and simply recomputes them on-the-fly during the backward pass when they are needed.

## Why This Matters

Memory is often the biggest bottleneck in deep learning. Checkpointing allows you to train significantly larger models or use larger batch sizes on a single GPU, making it a critical technique for scaling up.