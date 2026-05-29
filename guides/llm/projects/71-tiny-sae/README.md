# Tiny SAE

---

> Crack a layer's tangled activations into thousands of single-meaning features.

---

## Key Insight

This project trains a small [sparse autoencoder (SAE)](/shared/glossary/#sae) on the [residual stream](/shared/glossary/#residual-stream) of a small [LLM](/shared/glossary/#llm) and visualizes a handful of the recovered features — directions in activation space that fire for exactly one concept ("Golden Gate Bridge," "is a JSON key," "negation in a clause").

## Why This Matters

SAEs decompose a layer's dense, polysemantic activations into a much larger but mostly-zero set of [monosemantic](/shared/glossary/#monosemantic) features, giving the most promising current vocabulary for "what is the model thinking" — the working dictionary that drives interpretability research in 2024–2025.
