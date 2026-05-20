# Stride Explorer

Back to [Phase 1: Tensors and the Storage Model](../../README.md#phase-1-tensors-and-the-storage-model).

---

> "If you understand strides, you understand tensors. Everything else is commentary."

---

## What You Will Do

Print `.shape`, `.stride()`, `.storage_offset()`, and `.is_contiguous()` after every [`reshape`](/shared/glossary/#reshape), [`transpose`](/shared/glossary/#transpose), and [`permute`](/shared/glossary/#permute) on a few tensors. Watch how the numbers change — and notice what *doesn't* change.

## Key Insight

A [tensor](/shared/glossary/#tensor) is just a labeled window over a flat [storage](/shared/glossary/#storage) buffer: changing its [shape](/shared/glossary/#shape), [stride](/shared/glossary/#stride), or [offset](/shared/glossary/#offset) shifts how you read the same data — the data itself never moves.

## Why This Matters

Most surprising bugs in PyTorch — "why is my view failing?", "why did editing this tensor change another?" — come from not knowing the current stride layout. After this project, you will instinctively check strides whenever something feels wrong.

## Concepts to Watch

| Term | Plain-English Meaning |
|------|-----------------------|
| [Shape](/shared/glossary/#shape) | The grid size you see — e.g. "3 rows, 4 columns" |
| [Stride](/shared/glossary/#stride) | How many storage slots to jump when you move one step along a dimension |
| [Offset](/shared/glossary/#offset) | The starting slot inside storage where this tensor begins |
| [Contiguous](/shared/glossary/#contiguous) | The strides match the "natural" row-major order — no skipping around |

## A Real-Life Analogy

Imagine a hotel with 12 rooms numbered 1 to 12 in one long corridor. A "floor plan view" says the hotel has 3 floors × 4 rooms per floor. But the rooms haven't moved — the building just gave you a different *mental map*.

When you call `.transpose()`, PyTorch hands you a new mental map (new shape and stride) of the same corridor. Room 5 is still room 5 — you're just told to think of it as being on a different floor.

## Things to Try

1. Create `x = torch.arange(12).reshape(3, 4)`. Print all four attributes.
2. Call `y = x.t()` (transpose). Print again. Did the storage change?
3. Call `z = x.permute(1, 0)`. Is `z` the same as `y`? Check `is_contiguous()`.
4. Call `.contiguous()` on a non-contiguous tensor. Does `storage_offset()` reset?
5. Try `.reshape(2, 6)` on both `x` and `y`. Which one raises an error?

## Common Surprise

`.transpose()` and `.permute()` look like they rearrange data, but they only swap numbers in the stride tuple. The storage buffer is untouched. That is why transposing is almost free — and why the result can't always be passed to `.view()`.

## Difficulty

⭐ — great first project; takes about 30 minutes.
