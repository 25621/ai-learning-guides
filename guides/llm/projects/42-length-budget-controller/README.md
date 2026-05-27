# Length-Budget Controller

---

> Teach the model to think just long enough — and no longer.

---

## Key Insight

This project trains a model to obey an explicit "think for at most N tokens" instruction, then measures answer quality against the thinking budget it was given.

## Why This Matters

Longer reasoning costs more time and money, but more thinking is not always better. Controlling the budget lets a system spend [inference-time compute](/shared/glossary/#inference-time-compute) only where a hard problem actually needs it.
