# Mobile Deployment

---

> A phone is just another device — with the right runtime, your model runs there too.

---

## Key Insight

[ExecuTorch](/shared/glossary/#executorch) is PyTorch's runtime for phones and other edge devices. It takes a model captured by [`torch.export`](/shared/glossary/#torchexport) and runs it on hardware that is too small or too restricted to host full PyTorch.

## Why This Matters

Running a model directly on a device keeps data private, removes network delay, and works offline. ExecuTorch is the modern PyTorch path to get a model onto a phone.
