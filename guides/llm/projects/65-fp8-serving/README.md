# FP8 Serving

---

> Half the bits of bfloat16, much more throughput on modern GPUs.

---

## Key Insight

This project converts a model's weights, [activations](/shared/glossary/#activations), and [KV cache](/shared/glossary/#kv-cache) to [FP8](/shared/glossary/#fp8) using NVIDIA's TransformerEngine, verifies that quality on a small benchmark suite holds up, and measures the latency improvement on a Hopper-class GPU.

## Why This Matters

FP8 halves the memory and the bandwidth read for every parameter compared with [bfloat16](/shared/glossary/#bfloat16), and Hopper- and Blackwell-class GPUs have dedicated FP8 [Tensor Cores](/shared/glossary/#tensor-core), so the format is rapidly becoming the production default for new serving stacks — a near-free speedup when the hardware supports it.
