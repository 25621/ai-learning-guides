# Hardware Comparison

---

> The spec sheet predicts the benchmark — if you know which number to read.

---

## Key Insight

This project runs the same model and benchmark on two different GPUs and explains the performance gap directly from their spec sheets — especially their [HBM](/shared/glossary/#hbm) bandwidth and [TFLOPs](/shared/glossary/#tflops).

## Why This Matters

For [decode](/shared/glossary/#decode)-heavy chat, memory bandwidth decides the winner; for [prefill](/shared/glossary/#prefill)-heavy batch work, compute does. Learning to predict the benchmark from the spec sheet is what turns a hardware-buying decision from a guess into a defensible one.
