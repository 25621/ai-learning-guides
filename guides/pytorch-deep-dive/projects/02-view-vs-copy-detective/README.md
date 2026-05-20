# View vs Copy Detective

---

> "Debugging is twice as hard as writing the code in the first place."
> — Brian Kernighan
>
> With tensors, half of that debugging is figuring out whether you have a [view](/shared/glossary/#view) or a copy.

---

## Key Insight

A view shares [storage](/shared/glossary/#storage) with its source tensor, so editing through one shows up in the other — a copy owns independent storage, so edits stay private.

## Why This Matters

Suppose you slice a big tensor to train on a batch, then accidentally overwrite values through that slice. If the slice is a view, you have just corrupted your training data. This project builds the reflex to ask: "is this a view or a copy?" before modifying anything.
