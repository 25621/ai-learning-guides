# Triton Softmax

---

> Write a GPU kernel in Python — and watch it keep up with PyTorch's own.

---

## Key Insight

[Triton](/shared/glossary/#triton) lets you write a GPU [kernel](/shared/glossary/#kernel) in Python-like code instead of raw [CUDA](/shared/glossary/#cuda). Implementing [softmax](/shared/glossary/#softmax) — which reads a row, finds its max, exponentiates, and normalizes — and comparing it to `F.softmax` shows how close a hand-written kernel can get to PyTorch's built-in one.

## Why This Matters

Softmax is everywhere — every [attention](/shared/glossary/#attention) layer uses it — and writing it yourself in Triton is the gentlest on-ramp to GPU programming: no C++, no CUDA toolchain, just Python.
