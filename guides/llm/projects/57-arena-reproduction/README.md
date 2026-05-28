# Arena Reproduction

---

> Rank models the way chess ranks players: by who beats whom.

---

## Key Insight

This project runs a small tournament where five open models answer the same prompts, an [LLM-as-judge](/shared/glossary/#llm-as-judge) picks each winner, and the wins and losses become [Elo](/shared/glossary/#elo) ratings — the [arena](/shared/glossary/#arena) style of evaluation.

## Why This Matters

Pairwise "which is better?" comparisons are often more reliable than fixed-answer [benchmarks](/shared/glossary/#benchmark) for open-ended quality, but the resulting ranking can wobble with the random seed and the choice of judge — something you only appreciate by reproducing it.
