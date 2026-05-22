# Read native_functions.yaml

---

> Every operation PyTorch knows about is listed in one giant file — and you can read it.

---

## Key Insight

`native_functions.yaml` is the master list of every built-in PyTorch operation. Each entry declares an op's name and arguments and tells the [dispatcher](/shared/glossary/#dispatcher) which [kernel](/shared/glossary/#kernel) to run for each device and [dtype](/shared/glossary/#dtype).

## Why This Matters

This one file is the table of contents for [ATen](/shared/glossary/#aten). Knowing how to read it lets you discover exactly what an op does, which [backends](/shared/glossary/#backend) support it, and where its real implementation lives.
