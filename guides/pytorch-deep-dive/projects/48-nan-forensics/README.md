# NaN Forensics

---

> A single NaN doesn't stay put — it spreads to every number it touches.

---

## Key Insight

A [NaN](/shared/glossary/#nan) ("Not a Number") poisons every later calculation, so by the time training visibly breaks the real cause is many steps behind. Turning on [anomaly detection](/shared/glossary/#anomaly-detection) makes [autograd](/shared/glossary/#autograd) stop at the exact operation that first produced the NaN.

## Why This Matters

Chasing the symptom — a loss that suddenly reads NaN — wastes hours. Finding the first bad op tells you which layer and which math (a `log(0)`, a divide-by-zero, an exploding [gradient](/shared/glossary/#gradients)) to fix.
