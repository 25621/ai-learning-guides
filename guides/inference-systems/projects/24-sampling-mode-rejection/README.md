# Sampling-Mode Rejection

---

> Guessing ahead must not quietly change how random the model's answers are.

---

## Key Insight

This project extends speculative decoding to random [sampling](/shared/glossary/#sampling), not just greedy decoding: it adds a [rejection sampling](/shared/glossary/#rejection-sampling) step that accepts or rejects each draft token with exactly the right probability, so the final output is statistically identical to sampling from the [target model](/shared/glossary/#target-model) alone.

## Why This Matters

Without this careful accept/reject rule, speeding up generation would subtly distort the model's output distribution — making it more or less random than intended. The rejection step is what lets [speculative decoding](/shared/glossary/#speculative-decoding) stay provably lossless even with [temperature](/shared/glossary/#temperature) and [top-p](/shared/glossary/#top-p) sampling turned on.
