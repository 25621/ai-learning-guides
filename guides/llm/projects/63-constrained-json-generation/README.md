# Constrained JSON Generation

---

> Force the model to stay inside the schema's lines.

---

## Key Insight

This project uses [Outlines](/shared/glossary/#outlines) or [sglang](/shared/glossary/#sglang) to apply [constrained generation](/shared/glossary/#constrained-generation) at decode time — masking out any next-token choices that would break a target JSON schema — and measures the small overhead this masking adds compared to free, unconstrained generation.

## Why This Matters

The downstream tools and data pipelines that consume the model's reply — the JSON parser, the function-call router, the analytics job that loads the response into a database — cannot handle "almost-JSON," output that looks JSON-like but has a missing brace, a stray comma, or an unquoted key. Constraining the [sampling](/shared/glossary/#sampling) step to obey a schema guarantees structurally valid output on every call, which is the missing piece that makes [function calling](/shared/glossary/#function-calling) and tool-using agents reliable in production.
