# On-Device Build

---

> [Edge inference](/shared/glossary/#edge-inference) is the same KV-cache and quantization playbook — squeezed into a phone's battery and shared memory.

---

## Key Insight

This project compiles a 3B model to a real on-device runtime — Apple's MLX, TensorRT-LLM on Jetson, or a [GGUF](/shared/glossary/#gguf) build for `llama.cpp` — and measures actual tokens per second on the hardware in your hand. Running it on the device makes the constraints concrete: no data-center GPU, limited memory shared with the rest of the system, and a battery to respect.

## Why This Matters

[Edge inference](/shared/glossary/#edge-inference) is private, works offline, and costs nothing per request — but it lives or dies on fitting a [quantized](/shared/glossary/#quantization) model into a small memory and power envelope. The same [KV-cache](/shared/glossary/#kv-cache) and quantization principles from the rest of this guide carry over directly; this project shows how they feel when the "server" is a laptop or a phone.
