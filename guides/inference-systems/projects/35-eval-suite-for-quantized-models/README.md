# Eval-Suite for Quantized Models

---

> Never let a quantized model reach users without passing the same tests every time.

---

## Key Insight

This project builds an automated [quality gate](/shared/glossary/#quality-gate): a fixed set of evaluations — for example [perplexity](/shared/glossary/#perplexity) and capability tests like [MMLU](/shared/glossary/#mmlu) — that blocks a [quantized](/shared/glossary/#quantization) model from deploying if any one of them drops by more than an allowed amount.

## Why This Matters

Quantization regressions are silent — you usually won't notice until a user complains — so an automatic gate that fails the deploy on any meaningful score drop is the only reliable way to catch them before they reach production.
