# Activation Checkpointing Study

---

> Throw activations away on the way forward, recompute them on the way back — trade compute for memory.

---

## Key Insight

[Activation checkpointing](/shared/glossary/#activation-checkpointing) discards the intermediate [activations](/shared/glossary/#activations) saved during the forward pass and recomputes them during the [backward pass](/shared/glossary/#backward-pass). This study runs the same model with and without it, measuring the memory saved against the extra step time it costs.

## Why This Matters

For long sequences, activations — not [weights](/shared/glossary/#weights) — often dominate training memory. Spending a little recompute to reclaim a lot of memory is what lets you fit bigger batches or longer contexts onto the same GPU.
