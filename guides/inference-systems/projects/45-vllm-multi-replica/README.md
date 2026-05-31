# vLLM Multi-Replica

---

> When one copy isn't enough, run several and spread the load across them.

---

## Key Insight

This project runs four complete copies (replicas) of a model in [vLLM](/shared/glossary/#vllm) behind a simple round-robin [load balancer](/shared/glossary/#load-balancing) — a form of [data parallelism](/shared/glossary/#data-parallelism) — then load-tests them and reports the combined [throughput](/shared/glossary/#throughput).

## Why This Matters

Replication is the simplest way to scale: each copy is independent, so a failure stays isolated and capacity grows almost linearly with the number of replicas. It is the default choice for any model small enough to fit on a single GPU.
