# Module Introspection

---

> A model is a tree. Every node is a module, and every leaf is a parameter.

---

## Key Insight

[`nn.Module`](/shared/glossary/#nnmodule) is PyTorch's base class for all neural network components. When you assign `self.linear = nn.Linear(...)` inside `__init__`, the parent class notices and registers it. `named_modules()` walks this tree recursively, yielding every sub-module and its dot-separated path, so you can inspect any model without modifying it.

## Why This Matters

Knowing every layer's name, type, and parameter count is the foundation for debugging, profiling, and targeted fine-tuning. It is also the first step toward weight surgery: you must know the exact key names in a model before you can load or remap them.
