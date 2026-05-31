# Speculation + JSON-Mode

---

> When the output format is predictable, the model can guess most of it for free.

---

## Key Insight

This project adds prompt-lookup [speculative decoding](/shared/glossary/#speculative-decoding) — drafting next tokens by matching [n-grams](/shared/glossary/#n-gram) already seen in the prompt or schema — to a JSON-mode ([constrained generation](/shared/glossary/#constrained-generation)) workload, and measures the speedup. Because schema keys like `{"name":` are highly predictable, the draft is accepted in long runs and decoding jumps several tokens at a time.

## Why This Matters

Structured output is full of fixed, boilerplate spans that the model would otherwise emit one slow token at a time. Speculating those spans turns the most predictable part of generation into the fastest part, often giving dramatic speedups exactly on the tool-calling workloads that matter most in production.
