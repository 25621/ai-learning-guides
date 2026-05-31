# Speculation + JSON-Mode

---

> When the output format is predictable, the model can guess most of it for free.

---

## Key Insight

This project adds prompt-lookup [speculative decoding](/shared/glossary/#speculative-decoding) — drafting next tokens by matching [n-grams](/shared/glossary/#n-gram) already seen in the prompt or schema — to a JSON-mode ([constrained generation](/shared/glossary/#constrained-generation)) workload, and measures the speedup. Because schema keys like `{"name":` are highly predictable, the model's guesses (the draft) turn out right for many tokens in a row, so it can accept a whole chunk at once and leap several tokens forward in a single step instead of producing them one by one — like finishing a friend's sentence when you already know exactly how it ends.

## Why This Matters

Structured output is full of fixed, repeated filler — the same brackets, quotes, and key names every single time (the "boilerplate") — that the model would otherwise type out one slow token at a time, like hand-copying a form's pre-printed labels onto every new copy instead of just filling in the blanks. Speculating those spans turns the most predictable part of generation into the fastest part, often giving dramatic speedups exactly on the tool-calling workloads that matter most in production.
