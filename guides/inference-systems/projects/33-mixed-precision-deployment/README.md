# Mixed-Precision Deployment

---

> Keep the few fragile layers in full precision and squeeze everything else.

---

## Key Insight

This project leaves the most sensitive parts of the model — the [attention](/shared/glossary/#attention) output projections and the final output layer (`lm_head`) — in 16-bit [bfloat16](/shared/glossary/#bfloat16) while [quantizing](/shared/glossary/#quantization) everything else, then measures how much quality this recovers and what it costs in memory.

## Why This Matters

A handful of layers cause most of the quality loss when quantized, so spending extra bits only on them often recovers nearly all the accuracy for very little memory — a far better deal than quantizing every layer uniformly or leaving the whole model in full precision.
