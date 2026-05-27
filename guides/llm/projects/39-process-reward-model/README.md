# Process Reward Model

---

> Grade every step of the reasoning, not just the final answer.

---

## Key Insight

A [process reward model (PRM)](/shared/glossary/#process-reward-model) scores each individual step in a model's reasoning, unlike an [outcome reward model](/shared/glossary/#outcome-reward-model) that judges only the final answer. This project trains a small PRM on the PRM800K dataset and uses it to re-rank generated solutions.

## Why This Matters

Pinpointing the exact step where reasoning goes wrong gives a far sharper signal than a single right-or-wrong label on the final answer — which is why PRMs are strong scorers for [Best-of-N](/shared/glossary/#best-of-n) and tree search.
