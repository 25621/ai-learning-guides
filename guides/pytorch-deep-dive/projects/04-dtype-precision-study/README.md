# dtype Precision Study

---

> "Not all zeroes are created equal. Some of them used to be real numbers."

---

## Key Insight

A tensor's [dtype](/shared/glossary/#dtype) controls how each number is stored in memory: smaller formats use less memory and can be faster on modern GPUs, but they can silently discard tiny values — and in deep learning, tiny values often matter.

## Why This Matters

Most production training today uses `float16` or `bfloat16` for speed. If you don't understand their limits, you can end up with:
- Gradients that become zero (underflow) and stop learning
- Loss values that explode to `inf` or become `nan`
- Results that seem correct but are subtly wrong

This project makes you *see* those problems directly, before you encounter them in a real model.
