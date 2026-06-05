# Inference Optimization

## Key Insight

A trained [VLM](/shared/glossary/#vlm) is only useful if it can serve answers fast, so this project takes an open VLM and runs it on a production engine like [vLLM](/shared/glossary/#vllm) or [sglang](/shared/glossary/#sglang), then measures [throughput](/shared/glossary/#throughput) (tokens per second) as the number of images per request grows. Image count is the knob that matters because every image expands into many [image tokens](/shared/glossary/#token-visualaudio) that all live in the [KV cache](/shared/glossary/#kv-cache) and must be attended to — so more images means a longer sequence, more memory, and lower throughput, the multimodal twist on the usual long-context squeeze. The full serving toolkit (continuous batching, paged attention, quantization) is owned by the [Inference Systems](/guides/inference-systems/) guide; here the goal is just to *feel* how the image-token budget trades against speed.
