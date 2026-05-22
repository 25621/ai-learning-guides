# Trace One Op End to End

---

> When you call `torch.add`, Python is just the doorbell — the real work happens in C++.

---

## Key Insight

A Python call like `torch.add(a, b)` does no math itself. It travels through the [dispatcher](/shared/glossary/#dispatcher), which picks the right [kernel](/shared/glossary/#kernel) for your [tensor](/shared/glossary/#tensor)'s device and [dtype](/shared/glossary/#dtype), and lands in [ATen](/shared/glossary/#aten), the C++ library that does the actual arithmetic.

## Why This Matters

Once you can follow one operation from the Python call all the way down to its CPU kernel, the framework stops feeling like magic. You can then trace any op and answer questions the documentation never covers.
