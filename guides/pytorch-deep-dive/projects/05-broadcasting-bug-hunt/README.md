# Broadcasting Bug Hunt

---

> The most dangerous bugs are the ones that don't crash.
> Broadcasting bugs almost never crash.

---

## Key Insight

Broadcasting is *so* helpful that PyTorch will happily "stretch" tensors across mismatched shapes — and it will do so silently even when your shapes were just a typo away from what you meant.

## Why This Matters

A shape mismatch that *crashes* is easy to debug: you see the error, you fix the shape. A shape mismatch that *silently produces wrong values* can travel unnoticed through dozens of layers, corrupt a loss function, and only reveal itself as "the model isn't learning" — days later.
