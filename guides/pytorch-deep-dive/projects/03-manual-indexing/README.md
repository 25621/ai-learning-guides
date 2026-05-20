# Manual Indexing

Back to [Phase 1: Tensors and the Storage Model](../../README.md#phase-1-tensors-and-the-storage-model).

---

> The moment you can compute a flat storage index by hand is the moment tensors stop feeling like magic.

---

## What You Will Do

Given a tensor's [shape](/shared/glossary/#shape), [stride](/shared/glossary/#stride), and [offset](/shared/glossary/#offset), compute the flat [storage](/shared/glossary/#storage) index for `[i, j, k]` by hand — then check your answer with `.data_ptr()` in PyTorch.

## Key Insight

Multidimensional [indexing](/shared/glossary/#indexing) is just one formula: `flat_index = offset + i*stride[0] + j*stride[1] + k*stride[2]` — every PyTorch tensor operation is built on top of this arithmetic.

## Why This Matters

When you understand this formula, you can:
- Predict exactly why `.view()` fails on a non-contiguous tensor
- Understand why a sliced tensor and the original share memory
- Debug shape and stride bugs without guessing

## The Formula

For a tensor with `ndim` dimensions:

```
flat_index = offset
           + index[0] × stride[0]
           + index[1] × stride[1]
           + ...
           + index[ndim-1] × stride[ndim-1]
```

For a `(3, 4)` tensor with `stride=(4, 1)` and `offset=0`:
- Element `[1, 2]` → `0 + 1×4 + 2×1 = 6`
- Element `[2, 0]` → `0 + 2×4 + 0×1 = 8`

Try it with a transposed tensor — the strides swap and the formula still works perfectly.

## A Real-Life Analogy

Imagine a cinema with 5 rows × 10 seats. The theatre manager assigns each seat one unique number from 0 to 49:

```
seat_number = row × 10 + column
```

"Row 2, column 7" becomes seat 27. This is *exactly* what stride-based indexing does — except PyTorch also allows flexible row sizes (non-standard strides) and reserved front-row seats (non-zero offsets).

## Things to Try

1. Create `x = torch.arange(12).reshape(3, 4)`. Manually compute the flat index for `[1, 2]`. Confirm with `x.storage()[flat_index]`.
2. Compute the same for `y = x.t()`. Use `y.stride()` and `y.storage_offset()`.
3. Create a 3-D tensor `z = torch.arange(24).reshape(2, 3, 4)`. Compute the index for `[1, 1, 2]` by hand.
4. Slice `w = x[1:]`. What is `w.storage_offset()`? Verify that the formula still works.
5. Use [`data_ptr`](/shared/glossary/#data_ptr) to get the byte address of an element and confirm it matches your calculation (remember to multiply by bytes-per-element based on [dtype](/shared/glossary/#dtype)).

## A Helpful Mental Image

```
Storage:  [ 0  1  2  3  4  5  6  7  8  9  10  11 ]
           ↑
           offset=0

Tensor shape=(3,4), stride=(4,1):
  [0,0]=0  [0,1]=1  [0,2]=2  [0,3]=3
  [1,0]=4  [1,1]=5  [1,2]=6  [1,3]=7
  [2,0]=8  [2,1]=9  [2,2]=10 [2,3]=11

After .t() → shape=(4,3), stride=(1,4):
  [0,0]=0  [0,1]=4  [0,2]=8
  [1,0]=1  [1,1]=5  [1,2]=9
  ...same storage, different stride!
```

## Difficulty

⭐⭐ — takes about 1 hour; the "aha!" moment usually comes when you correctly predict the data_ptr result on a sliced or transposed tensor.
