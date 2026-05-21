# Custom Optimizer

---

> An optimizer is just a loop: read the gradient, update the state, move the parameter.

---

## Key Insight

Every PyTorch [optimizer](/shared/glossary/#optimizer) is a subclass of `torch.optim.Optimizer`. It holds a list of parameter groups and a per-parameter state dictionary. Its `step()` method reads each parameter's `.grad`, updates any running state (such as [momentum](/shared/glossary/#momentum)), and writes back a new value for the parameter.

## Why This Matters

Implementing [SGD](/shared/glossary/#sgd)-with-momentum from scratch shows you exactly where gradients flow after `loss.backward()`. Once you understand the pattern, you can implement any novel update rule — or debug why an existing one is misbehaving.
