# Quantize a 7B Model End-to-End

---

> Shrink the model to a quarter of its size — then prove it still answers just as well.

---

## Key Insight

This project takes a 7B model through the full serving-[quantization](/shared/glossary/#quantization) pipeline: pick [AWQ](/shared/glossary/#awq), [calibrate](/shared/glossary/#calibration) it on ~128 real-looking prompts, serve it with [vLLM](/shared/glossary/#vllm), and only ship it if it passes a [quality gate](/shared/glossary/#quality-gate).

## Why This Matters

Quantization is the biggest single lever on inference cost, but a careless one quietly degrades answers. Doing every step end-to-end — calibration *and* the gate — is how teams cut memory roughly 4× without secretly shipping a worse model.
