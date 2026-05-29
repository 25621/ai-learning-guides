# MMLU Re-run

---

> Before you trust a leaderboard number, reproduce it yourself.

---

## Key Insight

This project runs an open model through [MMLU](/shared/glossary/#mmlu) — a 57-subject multiple-choice [benchmark](/shared/glossary/#benchmark) — scores it per category, and checks the total against the number the model's authors published.

## Why This Matters

Reproducing a known score teaches you that small choices — the prompt format, the [sampling](/shared/glossary/#sampling) settings, how you parse the model's letter answer — can move a benchmark result by several points, so a single number means little without the setup that produced it.
