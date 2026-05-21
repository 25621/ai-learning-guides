# Bottleneck Fix

---

> Sometimes the compiler makes things slower — and the cure is to stop breaking its graph.

---

## Key Insight

[`torch.compile`](/shared/glossary/#torchcompile) is fastest on one unbroken graph. When it meets code it cannot trace — a `print`, a data-dependent branch, an unsupported op — it inserts a [graph break](/shared/glossary/#graph-break), splitting the model and falling back to slow [eager mode](/shared/glossary/#eager-mode) in between, sometimes making the whole run slower than not compiling at all.

## Why This Matters

Finding and removing graph breaks (with `TORCH_LOGS="graph_breaks"`) restores the speedup the compiler promised, and teaches you exactly which Python the compiler can and cannot handle — turning a [bottleneck](/shared/glossary/#bottleneck) back into a win.
