# The Indexing Formula

---

> Under the hood, every 5D tensor is just a 1D list.

---

## Key Insight

PyTorch uses a simple math formula to find data in memory: [`flat_index = offset + Σ(index * stride)`](/shared/glossary/#indexing). Mastering this formula demystifies how tensors actually work.

## Why This Matters

This formula is why PyTorch is fast. Knowing it helps you predict when operations will be efficient and why "non-contiguous" errors happen.
