# LR Schedule Sweep

---

> The learning rate is one number that changes over training — how you change it quietly decides where you land.

---

## Key Insight

A sweep trains the same model under different learning-rate schedules — cosine decay, [WSD](/shared/glossary/#wsd), and constant — and compares final [validation loss](/shared/glossary/#validation-loss) and downstream scores. The schedule controls how the learning rate rises during [warmup](/shared/glossary/#warmup) and falls afterward.

## Why This Matters

The schedule is one of the cheapest hyperparameters to get wrong and one of the highest-leverage to get right. Seeing the curves side by side builds intuition for why nearly every large run warms up then decays, and when the newer WSD recipe wins.
