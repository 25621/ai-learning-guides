# SLO Simulation

---

> "Find the exact load where your promise breaks — before your users find it for you."

---

## Key Insight

This project sets a concrete [SLO](/shared/glossary/#slo) — for example, p95 [TTFT](/shared/glossary/#ttft) under 500 ms — then steadily raises the request arrival rate until the system can no longer meet it, pinpointing the [bottleneck](/shared/glossary/#bottleneck) that gives out first.

## Why This Matters

Knowing the precise load at which your promise breaks turns a vague "it feels slow sometimes" into a number you can plan around — and tells you whether to fix the [scheduler](/shared/glossary/#scheduler), add GPUs, or shrink the model.
