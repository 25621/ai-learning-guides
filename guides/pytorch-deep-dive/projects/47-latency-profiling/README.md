# Latency Profiling

---

> Averages hide the slow requests — and the slow ones are what users remember.

---

## Key Insight

[Latency](/shared/glossary/#latency) is the time a single request takes. Reporting it as percentiles (p50, p95, p99) separates the typical request from the slow [tail](/shared/glossary/#tail-latency) that a plain average would hide. Larger batch sizes usually raise [throughput](/shared/glossary/#throughput) but also raise per-request latency.

## Why This Matters

Users feel the slow requests, not the average. Measuring p50/p95/p99 across batch sizes reveals the real latency–throughput trade-off, so you can pick a batch size that meets your latency target.
