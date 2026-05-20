# Custom Autograd Function

---

> Sometimes, you need to teach autograd new tricks.

---

## Key Insight

You are not limited to PyTorch's built-in operations. By subclassing `torch.autograd.Function`, you can define custom forward and [backward pass](/shared/glossary/#backward-pass) logic. You explicitly save required inputs using `ctx.save_for_backward()` and provide the exact derivative computation.

## Why This Matters

Custom autograd functions allow you to implement novel research ideas, optimize memory usage, or bypass non-differentiable steps. It is a powerful tool for bridging the gap between theoretical math and practical deep learning implementation.