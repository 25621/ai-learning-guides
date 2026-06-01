# Error Budget Tracker

---

> "Reliability is a balance you spend, not a box you check."

---

## Key Insight

An [SLO](/shared/glossary/#slo) of, say, 99.9% allows a small amount of failure; the leftover is the [error budget](/shared/glossary/#error-budget). This project computes the [SLI](/shared/glossary/#sli) — the measured success rate — each day and tracks how fast that budget is being spent under a chosen failure mode.

## Why This Matters

An error budget turns reliability into a currency: while budget remains you can ship risky changes, but once it is spent you freeze and stabilize. It replaces endless arguments about "is it reliable enough?" with a simple running balance.
