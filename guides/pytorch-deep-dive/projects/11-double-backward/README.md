# Double Backward

---

> Taking the gradient of a gradient.

---

## Key Insight

PyTorch's [autograd](/shared/glossary/#autograd) engine is capable of computing higher-order derivatives. By setting `create_graph=True` during the first [backward pass](/shared/glossary/#backward-pass), PyTorch tracks the gradient computation itself in a new [dynamic computation graph](/shared/glossary/#dynamic-computation-graph), allowing you to compute a [double backward](/shared/glossary/#double-backward) (the gradient of the gradient).

## Why This Matters

Higher-order derivatives are required for advanced techniques like gradient penalty in generative adversarial networks (GANs), meta-learning, and optimizing learning rates. Understanding double backward unlocks these cutting-edge optimization methods.