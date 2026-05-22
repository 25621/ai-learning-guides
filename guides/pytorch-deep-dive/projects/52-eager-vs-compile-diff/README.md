# Eager vs Compile Diff

---

> Same model, same input, two different answers — corner the one op that disagrees.

---

## Key Insight

[`torch.compile`](/shared/glossary/#torchcompile) fuses and reorders operations, which can change floating-point rounding compared to [eager mode](/shared/glossary/#eager-mode), so the two outputs may not match exactly. Bisecting the model layer by layer narrows the gap down to the single operation where they first diverge.

## Why This Matters

Most eager-vs-compiled differences are harmless rounding, but some signal a real compiler bug that quietly corrupts results. Isolating the exact op that differs is the only way to tell which case you are in and report it.
