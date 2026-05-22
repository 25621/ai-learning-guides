# Build PyTorch from Source

---

> You don't truly know a tool until you've compiled it yourself.

---

## Key Insight

PyTorch is a thin Python layer wrapped around a large C++ codebase — [ATen](/shared/glossary/#aten), [c10](/shared/glossary/#c10), and the [CUDA](/shared/glossary/#cuda) [kernels](/shared/glossary/#kernel). Building it from source compiles all of that C++ into the libraries that `import torch` loads.

## Why This Matters

A source build is the gateway to changing PyTorch itself. Doing it once — even though it is slow — turns the framework from a black box into code you can edit, patch, and explore.
