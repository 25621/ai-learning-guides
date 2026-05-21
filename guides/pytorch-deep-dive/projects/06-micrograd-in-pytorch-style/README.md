# Micrograd in PyTorch Style

---

> To understand autograd, build it yourself.

---

## Key Insight

PyTorch's [autograd](/shared/glossary/#autograd) is powered by a [dynamic computation graph](/shared/glossary/#dynamic-computation-graph) (DAG). Every time you perform an operation on a [tensor](/shared/glossary/#tensor) with `requires_grad=True`, PyTorch records it as a node in this graph. By recreating a simplified educational engine like [micrograd](/shared/glossary/#micrograd), you learn exactly how the forward pass builds the graph and how the [backward pass](/shared/glossary/#backward-pass) uses the [chain rule](/shared/glossary/#chain-rule) to calculate [gradients](/shared/glossary/#gradients).

## Why This Matters

It is easy to use `loss.backward()` as a magic black box, but understanding the underlying graph is the only way to debug [vanishing gradients](/shared/glossary/#vanishing-gradients), [detached tensors](/shared/glossary/#detached-tensor), and [memory leaks](/shared/glossary/#memory-leak) caused by holding onto graph references.