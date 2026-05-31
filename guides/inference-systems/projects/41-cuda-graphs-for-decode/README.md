# CUDA Graphs for Decode

---

> Record the dozens of tiny per-token kernels once, then replay them with almost no launch cost.

---

## Key Insight

This project captures the sequence of tiny [kernels](/shared/glossary/#kernel) that one [decode](/shared/glossary/#decode) step launches as a [CUDA Graph](/shared/glossary/#cuda-graphs), then replays that graph each step instead of launching the kernels one at a time.

## Why This Matters

Each decode step fires dozens of tiny kernels, and on small models the cost of *launching* them rivals the actual work. Replaying a captured graph removes most of that overhead for a 5–20% speedup.
