# Roofline Plot for Your Engine

---

> Every operating point is either starved for memory bandwidth or starved for compute — the plot tells you which.

---

## Key Insight

This project sweeps batch size and prompt length through your own inference engine and draws a [roofline](/shared/glossary/#roofline) plot — [throughput](/shared/glossary/#throughput) against [arithmetic intensity](/shared/glossary/#ai-arithmetic-intensity) — so you can see which operating points are memory-bound and which are compute-bound.

## Why This Matters

Whether to spend money on more [HBM](/shared/glossary/#hbm) bandwidth or more compute depends entirely on which side of the roofline your real workload sits. Measuring it yourself replaces guesswork with a picture you can point at.
