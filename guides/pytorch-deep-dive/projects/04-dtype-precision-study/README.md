# dtype Precision Study

Back to [Phase 1: Tensors and the Storage Model](../../README.md#phase-1-tensors-and-the-storage-model).

---

> "Not all zeroes are created equal. Some of them used to be real numbers."

---

## What You Will Do

Compare [`float32`](/shared/glossary/#float32), [`float16`](/shared/glossary/#float16), and [`bfloat16`](/shared/glossary/#bfloat16) by summing one million small numbers and observing how much precision each format loses — and when values disappear entirely through [underflow](/shared/glossary/#underflow).

## Key Insight

A tensor's [dtype](/shared/glossary/#dtype) controls how each number is stored in memory: smaller formats use less memory and can be faster on modern GPUs, but they can silently discard tiny values — and in deep learning, tiny values often matter.

## Why This Matters

Most production training today uses `float16` or `bfloat16` for speed. If you don't understand their limits, you can end up with:
- Gradients that become zero (underflow) and stop learning
- Loss values that explode to `inf` or become `nan`
- Results that seem correct but are subtly wrong

This project makes you *see* those problems directly, before you encounter them in a real model.

## The Three Formats at a Glance

| Format | Bits | Approx. range | Approx. smallest non-zero |
|--------|------|---------------|---------------------------|
| [`float32`](/shared/glossary/#float32) | 32 | ±3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ |
| [`bfloat16`](/shared/glossary/#bfloat16) | 16 | ±3.4 × 10³⁸ | ~1.2 × 10⁻³⁸ |
| [`float16`](/shared/glossary/#float16) | 16 | ±65,504 | ~6 × 10⁻⁸ |

`bfloat16` keeps `float32`'s *range* but cuts precision. `float16` keeps better precision for moderate-sized values but has a much smaller range — which is exactly where underflow lurks.

## A Real-Life Analogy

Imagine keeping a budget spreadsheet, but your app only stores amounts rounded to the nearest $100. If you have a $1 expense, it rounds to $0 and vanishes. Now multiply that across a million gradient updates in a neural network.

`float16` is that app: powerful and compact, but small numbers fall off the edge. `bfloat16` is a better app for training: it rounds, but almost nothing falls completely off.

## Things to Try

1. Create `values = torch.full((1_000_000,), 0.0001)`. Sum in `float32`, `float16`, and `bfloat16`. Compare results.
2. Create `tiny = torch.tensor(1e-8, dtype=torch.float16)`. What does PyTorch store? Check with `.item()`.
3. Compute `1.0 + 1e-7` in each dtype. Which formats lose the small increment?
4. Try summing in a loop (Python `for` loop, not `.sum()`) in `float16`. Does the running total freeze at some point?
5. Look up `torch.finfo(torch.float16).tiny` and `torch.finfo(torch.bfloat16).tiny`. Can you construct a value that underflows each format?

## The Underflow Moment

```python
x = torch.tensor(1e-8, dtype=torch.float16)
print(x)  # tensor(0., dtype=torch.float16)  ← it became zero!
```

That value was real. `float16` could not hold it and silently replaced it with zero. In a gradient, that silent zero means: "stop learning this weight." The model will never tell you why it got stuck.

## Difficulty

⭐⭐ — takes about 1 hour; the most valuable part is step 4, where you watch a running sum freeze in real time.
