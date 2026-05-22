# Patch and Rebuild

---

> The fastest way to believe you can change PyTorch is to watch your own `printf` fire.

---

## Key Insight

Adding a simple `printf` inside a [CUDA](/shared/glossary/#cuda) [kernel](/shared/glossary/#kernel) and rebuilding lets you confirm that the code you just edited is the same code that runs when you call the op from Python.

## Why This Matters

This tiny change proves the whole loop — edit C++, recompile, run from Python — works on your machine. After that, real fixes and experiments are just bigger versions of the same loop.
