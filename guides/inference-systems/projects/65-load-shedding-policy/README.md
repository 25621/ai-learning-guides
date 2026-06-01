# Load-Shedding Policy

---

> "When you can't serve everyone well, protect the requests that matter most."

---

## Key Insight

Under overload, [load shedding](/shared/glossary/#load-shedding) deliberately refuses some requests early — using priority-aware [admission control](/shared/glossary/#admission-control) — so the most important traffic still meets its [SLO](/shared/glossary/#slo). This project verifies that high-priority requests stay fast even at 2× overload.

## Why This Matters

When demand exceeds capacity, trying to serve everyone degrades the experience for everyone. Dropping low-priority work quickly — a fast rejection beats a slow timeout — keeps the requests that matter within their targets.
