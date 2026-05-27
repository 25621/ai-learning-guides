# Train a Reward Model

---

> Turn "humans liked this answer better" into a number a machine can chase.

---

## Key Insight

This project trains a [reward model](/shared/glossary/#reward-model) on human preference pairs — for a given prompt, which of two answers a person preferred — and reports how often it agrees with people. The model learns to give a higher score to the response a human would have picked.

## Why This Matters

A reward model converts messy, one-off human judgment into a reusable score that [RLHF](/shared/glossary/#rlhf) can optimize against millions of times. Its accuracy sets a ceiling on how good the aligned model can ever become.
