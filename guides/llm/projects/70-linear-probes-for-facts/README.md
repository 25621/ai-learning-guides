# Linear Probes for Facts

---

> Open the black box one layer at a time and ask "where in here does the model know X?"

---

## Key Insight

This project trains a [linear probe](/shared/glossary/#linear-probe) — a single linear classifier — on the hidden [activations](/shared/glossary/#activations) at each [transformer](/shared/glossary/#transformer) layer to recover whether a statement is factually true, mapping out which layers actually encode that knowledge.

## Why This Matters

A working probe is a window into [mechanistic interpretability](/shared/glossary/#mechanistic-interpretability): it shows that the model carries the factuality signal internally before it generates its answer, which is the starting point for explaining *why* and *where* a behavior happens inside the network.
