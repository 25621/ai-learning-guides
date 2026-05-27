# Compute Calculator

---

> Six times parameters times tokens — the whole cost of a training run on the back of an envelope.

---

## Key Insight

A forward-and-backward pass over a dense [transformer](/shared/glossary/#transformer) with `N` [parameters](/shared/glossary/#parameters) on `D` tokens costs roughly `6 N D` [FLOPs](/shared/glossary/#flops). This project implements that one formula and checks its prediction against a real run's measured wall-time and FLOPs.

## Why This Matters

Being able to estimate training compute in your head turns a vague worry ("can I afford this run?") into simple arithmetic. It is the first sanity check you do before committing a single GPU-hour.
