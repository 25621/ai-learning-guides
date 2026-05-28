# Quantize a 7B Model

---

> Smaller numbers, the same model, a fraction of the memory.

---

## Key Insight

This project takes a 7B model and applies two [quantization](/shared/glossary/#quantization) recipes — [GPTQ](/shared/glossary/#gptq) to compress weights to [INT8](/shared/glossary/#int8) and [AWQ](/shared/glossary/#awq) to compress them to 4-bit — then measures how much GPU memory is saved and how a few [benchmark](/shared/glossary/#benchmark) scores change.

## Why This Matters

Shrinking weights from 16-bit down to 8- or 4-bit can fit a 7B model on a consumer GPU and read it through memory faster, which is the simplest way to cut both the cost and the latency of serving while usually paying only a small quality cost.
