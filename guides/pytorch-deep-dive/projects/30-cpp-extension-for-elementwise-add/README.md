# C++ Extension for Elementwise Add

---

> When Python is too slow, drop into C++ — and PyTorch will still treat it like a built-in op.

---

## Key Insight

A [C++ extension](/shared/glossary/#c-extension) lets you write an operation in C++ (or [CUDA](/shared/glossary/#cuda)), compile it, and call it from Python as if it were built in. Writing an [elementwise](/shared/glossary/#elementwise-operation) `add_cuda`, registering it, and calling it shows the full path a call travels — from Python, through the [dispatcher](/shared/glossary/#dispatcher), down to a compiled [kernel](/shared/glossary/#kernel).

## Why This Matters

This is your escape hatch when an operation is missing or too slow. A custom extension is exactly how new ops enter PyTorch, so walking the path once makes the framework feel less like a black box.
