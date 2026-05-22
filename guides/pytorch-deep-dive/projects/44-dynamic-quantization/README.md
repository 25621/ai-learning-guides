# Dynamic Quantization

---

> Store the weights as 8-bit integers and decide the activation scale on the fly.

---

## Key Insight

[Quantization](/shared/glossary/#quantization) stores a model's [weights](/shared/glossary/#weights) in low-precision integers like [int8](/shared/glossary/#int8) instead of 32-bit floats. [Dynamic quantization](/shared/glossary/#dynamic-quantization) keeps the weights quantized ahead of time but computes the scale for each layer's [activations](/shared/glossary/#activations) at runtime, just before the [matmul](/shared/glossary/#matmul).

## Why This Matters

int8 weights use a quarter of the memory and run faster on many CPUs, which helps most with the large linear layers in an [LLM](/shared/glossary/#llm). Measuring the quality drop tells you whether the speedup is worth it.
