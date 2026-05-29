# Diffusion vs LLM Serving

---

> An image model has no "first token" to stream — and that one fact reshapes the whole serving stack.

---

## Key Insight

This project puts a [Stable-Diffusion-style](/shared/glossary/#diffusion-model) image model and a 7B [LLM](/shared/glossary/#llm) behind the same load generator and contrasts the two shapes side by side. The LLM has a short [prefill](/shared/glossary/#prefill) followed by a long stream of one-token [decode](/shared/glossary/#decode) steps that grow a [KV cache](/shared/glossary/#kv-cache); the diffusion model has none of that — it runs many fixed-shape denoising passes that look like prefill repeated, all returned at once when the image is done.

## Why This Matters

The same control knobs do not work on both workloads: continuous batching, prefix caching, and speculative decoding are LLM-specific levers that have no diffusion analog, while step distillation and CFG fusion are diffusion-specific levers that have no LLM analog. Trying to serve both with one engine, or copy-pasting tuning advice from one to the other, is a common and expensive mistake that this comparison makes obvious.
