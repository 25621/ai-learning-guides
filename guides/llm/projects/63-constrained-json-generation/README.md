# Constrained JSON Generation

---

> Force the model to stay inside the schema's lines.

---

## Key Insight

This project uses Outlines or sglang to apply [constrained generation](/shared/glossary/#constrained-generation) at decode time — masking out any next-token choices that would break a target JSON schema — and measures the small overhead this masking adds compared to free, unconstrained generation.

## Why This Matters

Downstream tools and pipelines cannot parse "almost-JSON," so constraining the [sampling](/shared/glossary/#sampling) step to obey a schema guarantees structurally valid output on every call — the missing piece that makes [function calling](/shared/glossary/#function-calling) and tool-using agents reliable in production.
