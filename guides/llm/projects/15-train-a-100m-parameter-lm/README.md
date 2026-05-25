# Train a 100M-Parameter LM

---

> Ten times bigger, real web text, and a number to beat — the first run that feels like the real thing.

---

## Key Insight

Scaling to 100 million [parameters](/shared/glossary/#parameters) and training on a slice of real web text (OpenWebText) is the first [pretraining](/shared/glossary/#pretraining) run that behaves like a production one. The concrete goal — push [validation loss](/shared/glossary/#validation-loss) below 3.5 — turns "is it working?" into a measurable target.

## Why This Matters

Real data, a real GPU, and a fixed time budget force the skill every practitioner needs: reading a [loss](/shared/glossary/#loss-function) curve to judge whether a run is healthy, stalled, or diverging — long before it finishes.
