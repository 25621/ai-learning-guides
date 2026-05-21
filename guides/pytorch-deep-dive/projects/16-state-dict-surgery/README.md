# State Dict Surgery

---

> A model is just a dictionary. Knowing the keys is everything.

---

## Key Insight

A [state dict](/shared/glossary/#state-dict) is a plain Python ordered dictionary that maps parameter and buffer names to their tensor values. Loading weights into a different architecture is a matter of matching these key names and shapes — whether by renaming keys, slicing tensors, or ignoring mismatches with `strict=False`.

## Why This Matters

Real-world models are rarely loaded from a checkpoint with identical architecture. Transfer learning, model merging, and checkpoint recovery all require mapping weights between differently structured models. State dict surgery is the practical skill that makes all of these possible.
