# SLO-Aware Scheduler

---

> Meeting the deadline matters more than being fast on average.

---

## Key Insight

This project builds a [scheduler](/shared/glossary/#scheduler) that knows each request's deadline — its [SLO](/shared/glossary/#slo) — and orders work to finish as many requests on time as possible, then compares it against plain [first-come, first-served](/shared/glossary/#fcfs) ordering.

## Why This Matters

A good average speed can still hide many missed deadlines. A deadline-aware scheduler can complete far more requests within their [latency](/shared/glossary/#latency) targets than naive ordering — which is exactly what users notice and what service contracts are written against.
