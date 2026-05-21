# The Precision Trade-off

---

> Smaller numbers are faster, but they can't always tell the truth.

---

## Key Insight

A tensor's [dtype](/shared/glossary/#dtype) determines its memory size and accuracy. Low-precision types (like `float16`) save space but can "round off" important small values to zero.

## Why This Matters

In deep learning, small values (like gradients) are everything. Choosing the wrong precision can cause your model to stop learning or explode with "NaN" errors.
