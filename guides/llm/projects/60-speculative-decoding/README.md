# Speculative Decoding

---

> Let a small model guess and the big one only check.

---

## Key Insight

This project pairs a 1B draft model with a 7B target model for [speculative decoding](/shared/glossary/#speculative-decoding): the draft cheaply proposes a few tokens ahead at each step, the target verifies them all in a single parallel pass, and the longest matching prefix is kept. The project then measures the acceptance rate and the wall-clock speedup.

## Why This Matters

Single-token decoding is the slowest part of serving an [LLM](/shared/glossary/#llm) because it cannot be batched along the time axis — each new token has to wait for the one before it to be sampled before it can even start, like cars in a single-lane tunnel where the next car cannot enter until the one ahead clears the gate. (Batching *across* requests adds more cars in parallel lanes; batching *along the time axis* would mean letting cars in the same lane move at once, which the chain of dependencies forbids.) Letting a cheap helper propose several tokens that the big model verifies in parallel sidesteps this bottleneck and routinely gives a 2–4× speedup with no quality loss, which is why speculative decoding is now standard in production stacks.
