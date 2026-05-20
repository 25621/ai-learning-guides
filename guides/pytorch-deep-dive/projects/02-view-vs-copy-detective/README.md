# View vs Copy Detective

Back to [Phase 1: Tensors and the Storage Model](../../README.md#phase-1-tensors-and-the-storage-model).

---

> "Debugging is twice as hard as writing the code in the first place."
> — Brian Kernighan
>
> With tensors, half of that debugging is figuring out whether you have a [view](/shared/glossary/#view) or a copy.

---

## What You Will Do

1. Modify a tensor through a [view](/shared/glossary/#view) and observe whether the original tensor changes.
2. Find the operations that silently make a [copy](/shared/glossary/#copy) instead of a view.

## Key Insight

A view shares [storage](/shared/glossary/#storage) with its source tensor, so editing through one shows up in the other — a copy owns independent storage, so edits stay private.

## Why This Matters

Suppose you slice a big tensor to train on a batch, then accidentally overwrite values through that slice. If the slice is a view, you have just corrupted your training data. This project builds the reflex to ask: "is this a view or a copy?" before modifying anything.

## The Central Question

Two tensors can look identical but behave very differently:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = a.view(3)      # same storage
c = a.clone()      # new storage

b[0] = 99.0
print(a[0])        # 99.0  — a and b share storage
print(c[0])        # 1.0   — c is its own copy
```

Use `a.data_ptr() == b.data_ptr()` to check whether two tensors share the same starting address in memory.

## Operations to Investigate

| Operation | Usually a view? | Notes |
|-----------|-----------------|-------|
| [`view`](/shared/glossary/#view) | ✅ Always | Fails if the tensor is not [contiguous](/shared/glossary/#contiguous) |
| [`reshape`](/shared/glossary/#reshape) | ✅ When possible | Falls back to a copy if needed — you cannot always tell which |
| [`permute`](/shared/glossary/#permute) | ✅ Always | Only swaps strides |
| [`transpose`](/shared/glossary/#transpose) | ✅ Always | Same as permute for two dims |
| `.clone()` | ❌ Always a copy | Explicit, safe |
| `.contiguous()` | ❌ Copies if needed | View if already contiguous |

## A Real-Life Analogy

Think of a shared Google Doc. Two colleagues can have the document open at the same time — if one person edits a paragraph, the other immediately sees the change. That is a view.

Now imagine one colleague downloads the file to their laptop and works offline. Changes on their laptop do not affect the original until they upload again. That is a copy.

PyTorch views work like the shared doc: fast, memory-efficient, but surprising if you forget someone else is watching.

## Things to Try

1. Create `a = torch.arange(6.0).reshape(2, 3)`. Assign `b = a[0]`. Modify `b[0]`. Did `a` change?
2. Try `c = a.permute(1, 0)`. Modify `c[0, 0]`. What happens to `a[0, 0]`?
3. Use `reshape` to get a non-contiguous tensor, then reshape it again. Did you get a view or a copy? Check with `data_ptr`.
4. Use `.clone()` and confirm that edits to the clone do not touch the original.
5. Find one operation that *looks* like a view but is documented as always copying.

## The Detective's Checklist

When you are unsure whether two tensors share storage:
- Compare `.data_ptr()` — same address = shared storage
- Check `.is_contiguous()` — non-contiguous layouts are more likely to force a copy on the next operation
- Read the docs for the operation — look for phrases like "returns a view" or "may copy"

## Difficulty

⭐⭐ — takes about 45 minutes; the tricky part is finding where `reshape` silently copies.
