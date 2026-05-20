# View vs. Copy Detective

---

> Is it a new tensor, or just a new way of looking at the old one?

---

## Key Insight

A [view](/shared/glossary/#view) shares memory with its parent; a copy is independent. If you change a view, you change the original [storage](/shared/glossary/#storage).

## Why This Matters

Accidentally modifying a view can corrupt your original dataset or model weights. Learning to spot the difference prevents these silent, hard-to-find bugs.
