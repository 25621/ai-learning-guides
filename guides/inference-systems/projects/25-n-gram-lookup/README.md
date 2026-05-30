# N-gram Lookup

---

> When the answer is already sitting in the prompt, just copy it forward.

---

## Key Insight

This project implements prompt-lookup decoding: instead of a [draft model](/shared/glossary/#draft-model), it scans the prompt for a matching [n-gram](/shared/glossary/#n-gram) — a short run of recent tokens — and reuses whatever text followed it last time as the draft guess. It needs no extra model and no training.

## Why This Matters

When the model is mostly copying from the prompt — summarizing, editing code, or answering from a retrieved document ([RAG](/shared/glossary/#rag)) — the next tokens often already appear earlier in the text, so this free trick gives large speedups exactly where training a separate [draft model](/shared/glossary/#draft-model) would be overkill.
