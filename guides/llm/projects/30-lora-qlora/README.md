# LoRA / QLoRA

---

> Fine-tune a billion-parameter model without renting a cluster.

---

## Key Insight

This project repeats [SFT](/shared/glossary/#sft) with [LoRA](/shared/glossary/#lora) adapters, then with [QLoRA](/shared/glossary/#qlora) — which adds 4-bit [quantization](/shared/glossary/#quantization) — and compares quality and memory use against a full fine-tune. Instead of updating all the [weights](/shared/glossary/#weights), you train a small set of extra low-rank matrices and keep the original model frozen.

## Why This Matters

LoRA and QLoRA are what let you fine-tune a multi-billion-parameter model on a single consumer GPU. They turn customizing large models from a datacenter job into something anyone can do on one card.
