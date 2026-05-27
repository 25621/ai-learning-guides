# Long-Context Extension

---

> A model trained on short text can often be stretched to long text by rescaling how it counts position.

---

## Key Insight

A model trained at a 4k [context window](/shared/glossary/#context-window) can be extended to longer inputs by rescaling its [RoPE](/shared/glossary/#rope) angles — via [position interpolation](/shared/glossary/#position-interpolation) or [YaRN](/shared/glossary/#yarn) — usually with little or no retraining.

## Why This Matters

Pretraining at long context is expensive, so most long-context models are extended after the fact. Testing the result with a [needle-in-a-haystack](/shared/glossary/#needle-in-a-haystack) probe shows whether the model truly uses the new length or just tolerates it.
