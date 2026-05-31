# JSON-Mode Reliability

---

> When the model literally cannot emit an invalid character, invalid output drops to zero.

---

## Key Insight

This project runs the same prompt on the same model 1000 times with and without [constrained generation](/shared/glossary/#constrained-generation), then compares how often the output is valid JSON. Without constraints a small model fails a few percent of the time; with constraints — masking out every token that would break the schema before sampling — the failure rate falls to essentially zero.

## Why This Matters

Tool-calling and agent pipelines break the moment a model emits a stray comma or an unclosed brace, and these failures are random and hard to debug. Forcing structurally valid output at decode time is the cheapest, most reliable way to make an LLM safe to wire into downstream software.
