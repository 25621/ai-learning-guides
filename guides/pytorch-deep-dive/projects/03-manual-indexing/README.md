# Manual Indexing

---

> The moment you can compute a flat storage index by hand is the moment tensors stop feeling like magic.

---

## Key Insight

Multidimensional [indexing](/shared/glossary/#indexing) is just one formula: `flat_index = offset + i*stride[0] + j*stride[1] + k*stride[2]` — every PyTorch tensor operation is built on top of this arithmetic.

## Why This Matters

When you understand this formula, you can:
- Predict exactly why `.view()` fails on a non-contiguous tensor
- Understand why a sliced tensor and the original share memory
- Debug shape and stride bugs without guessing
