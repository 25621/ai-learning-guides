# Broadcasting Bug Hunt

Back to [Phase 1: Tensors and the Storage Model](../../README.md#phase-1-tensors-and-the-storage-model).

---

> The most dangerous bugs are the ones that don't crash.
> Broadcasting bugs almost never crash.

---

## What You Will Do

Construct five tensor expressions where [broadcasting](/shared/glossary/#broadcasting) runs without an error but produces a result you did not intend. For each one, write down exactly which rule you violated.

## Key Insight

Broadcasting is *so* helpful that PyTorch will happily "stretch" tensors across mismatched shapes — and it will do so silently even when your shapes were just a typo away from what you meant.

## Why This Matters

A shape mismatch that *crashes* is easy to debug: you see the error, you fix the shape. A shape mismatch that *silently produces wrong values* can travel unnoticed through dozens of layers, corrupt a loss function, and only reveal itself as "the model isn't learning" — days later.

## How Broadcasting Works

PyTorch aligns shapes from the *right* and applies these rules dimension by dimension:
1. If both dimensions are the same size → OK, no stretching needed.
2. If one dimension is 1 → the size-1 dimension is stretched to match the other.
3. If neither dimension is 1 and they differ → **error**.

```python
# Shape (3,) + (4, 3) → OK: (3,) is treated as (1, 3), stretched to (4, 3)
a = torch.ones(3)
b = torch.ones(4, 3)
(a + b).shape   # → (4, 3) ✅

# Shape (4,) + (4, 3) → ERROR: 4 ≠ 3, neither is 1
c = torch.ones(4)
(c + b).shape   # RuntimeError ❌
```

## A Real-Life Analogy

Broadcasting is like an autocorrect that fixes some typos automatically. Most of the time it is helpful. But occasionally it "corrects" a word you meant to type exactly as you typed it, and the sentence now means something completely different — while looking perfectly grammatical.

The sentence passed spell-check. It is still wrong.

## Bug Categories to Hunt

Try to find one example from each category:

| Category | Example Setup |
|----------|--------------|
| **Wrong axis** | You meant to add a bias per row, but you add it per column instead |
| **Missing dimension** | A `(batch,)` tensor broadcasts against a `(batch, seq, hidden)` tensor in an unexpected way |
| **Accidental outer product** | `(4, 1) * (1, 3)` gives `(4, 3)` — did you mean element-wise? |
| **Transposed shapes** | `(3, 1)` vs `(1, 3)` — both broadcast, but they give very different results |
| **Summing in wrong direction** | After a broadcasting op, the [shape](/shared/glossary/#shape) is larger than you expected; summing `.sum()` collapses too much |

## Things to Try

1. Add a `(3,)` bias to a `(4, 3)` matrix. Then accidentally swap the bias to shape `(4,)` — what does the result look like?
2. Multiply a `(4, 1)` column vector by a `(1, 4)` row vector. What shape comes out? Is that what you wanted?
3. Create a `(2, 3, 4)` [tensor](/shared/glossary/#tensor) and add a `(4,)` tensor. Now add a `(3, 1)` tensor. Compare the two results.
4. Try to add a `(4,)` and a `(3,)` tensor. What happens? Now add `(4, 1)` and `(1, 3)` — what changes?
5. For each bug you found: write one sentence describing the rule you violated. This is the most important step.

## The Rule Worth Memorizing

> Shapes align from the **right**. A dimension of size `1` is a wildcard. Everything else must match exactly.

When a broadcast operation succeeds but the result is larger than you expected, that is the moment to stop and question your shapes.

## Difficulty

⭐⭐ — takes about 1 hour; writing the rule you violated (the last step) is harder than producing the bug and is also more valuable.
