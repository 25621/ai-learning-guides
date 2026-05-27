# Best-of-N with a Reward Model

---

> Generate many answers, then keep the one a judge likes best.

---

## Key Insight

[Best-of-N](/shared/glossary/#best-of-n) sampling draws `N` candidate answers and uses a [reward model](/shared/glossary/#reward-model) to score them, keeping the highest-scoring one. This project compares that approach against [self-consistency](/shared/glossary/#self-consistency) majority voting on a math benchmark.

## Why This Matters

When a learned scorer recognizes a good answer better than a plain vote does, Best-of-N picks winners that majority voting would miss. It is a simple, effective way to spend extra inference compute on quality.
