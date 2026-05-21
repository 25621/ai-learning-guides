# Reproducible Training

---

> Same seed, same model, same data — same number, every time.

---

## Key Insight

Full reproducibility requires more than setting a random seed. You also need to enable [deterministic algorithms](/shared/glossary/#deterministic-algorithms) in PyTorch (via `torch.use_deterministic_algorithms(True)`), set seeds for Python, NumPy, and CUDA, and control data order with a fixed DataLoader seed.

## Why This Matters

When a training run gives unexpected results, reproducibility lets you bisect the problem: run it twice, compare the outputs, and confirm whether the behavior is deterministic. Without it, debugging is guesswork.
