# Custom Op Registration

---

> Register your kernel as a real op, and the compiler will treat it as one of its own.

---

## Key Insight

Wrapping your [kernel](/shared/glossary/#kernel) as a [custom op](/shared/glossary/#custom-op) with `torch.library.custom_op` gives PyTorch the metadata it needs (such as the output shape) to treat your code as a first-class operation. Without this, [`torch.compile`](/shared/glossary/#torchcompile) hits an op it cannot trace and inserts a [graph break](/shared/glossary/#graph-break); with it, your kernel stays inside the optimized graph.

## Why This Matters

A fast custom kernel is far less useful if it forces the compiler to stop, so proper registration is what lets hand-written and compiled code work together instead of fighting each other.
