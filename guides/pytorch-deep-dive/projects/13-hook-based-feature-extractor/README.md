# Hook-Based Feature Extractor

---

> You do not need to rewrite a model to see inside it.

---

## Key Insight

A [forward hook](/shared/glossary/#forward-hook) is a callback you register on any `nn.Module`. PyTorch calls it automatically after that module's forward pass, passing in the input and output tensors. You can capture the output — called [activations](/shared/glossary/#activations) — without touching the model's code at all.

## Why This Matters

Feature extraction and visualization are essential for understanding what a network has learned. Hooks let you tap into any layer of any pretrained model in just a few lines, making them the standard tool for interpretability and transfer learning.
