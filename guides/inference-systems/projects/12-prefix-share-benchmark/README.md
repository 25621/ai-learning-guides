# Prefix-Share Benchmark

---

> If a thousand requests all start with the same system prompt, you should only pay to read it once.

---

## Key Insight

This project sends many requests that share a long [system prompt](/shared/glossary/#system-prompt) to [vLLM](/shared/glossary/#vllm) and measures [time to first token](/shared/glossary/#ttft) with the [prefix cache](/shared/glossary/#prefix-cache) turned on versus off. When requests share an opening, the engine can reuse the already-computed cache for those tokens instead of redoing the work.

## Why This Matters

Production chat apps reuse the same long system prompt on every request, so prefix caching can cut time to first token several-fold for free. Measuring the before/after yourself shows how large these "shared prefix" wins are and why engines build whole data structures just to capture them.
