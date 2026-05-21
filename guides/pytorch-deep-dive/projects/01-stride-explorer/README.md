# Stride Explorer

---

> Tensors don't store grids; they store lines and use "strides" to jump through them.

---

## Key Insight

A [tensor](/shared/glossary/#tensor) is a view into a flat [storage](/shared/glossary/#storage) buffer. By changing [strides](/shared/glossary/#stride), PyTorch can "re-shape" data instantly without moving a single byte.

## Why This Matters

Understanding strides explains why some operations (like `.view()`) fail and why others (like `.transpose()`) are free. It is the secret to writing fast, memory-efficient code.
