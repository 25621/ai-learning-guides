# Request-Lifecycle Tracer

---

> You cannot fix the slow request you cannot see.

---

## Key Insight

This project attaches a timestamp to each stage a request passes through — admission, queue wait, [prefill](/shared/glossary/#prefill), each [decode](/shared/glossary/#decode) step, detokenization, send — and produces a flamegraph for the slowest request from a load test. The picture immediately reveals which stage owns the [tail latency](/shared/glossary/#tail-latency): a long queue wait points at the scheduler, a long prefill points at prompt length or [chunked-prefill](/shared/glossary/#chunked-prefill) tuning, a long decode tail points at memory bandwidth.

## Why This Matters

Production [SLOs](/shared/glossary/#slo) are almost always quoted at the tail (P99), and the tail is a different request than the median — fixing the average does nothing for the user whose reply took ten seconds. Per-stage tracing is the cheapest way to keep that user in view, and a working tracer is a prerequisite for any later performance work in this guide.
