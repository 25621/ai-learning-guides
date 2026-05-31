# Failure-Mode Drill

---

> Pull the plug on a replica under load and see whether users even notice.

---

## Key Insight

This project kills one replica in the middle of a load test and checks that the system fails over gracefully — rerouting traffic to the healthy replicas. You measure the user-visible impact straight from the load generator's own numbers: how many requests came back as errors (an HTTP 5xx or a dropped connection), how far [latency](/shared/glossary/#latency) spiked, and how many seconds it took to recover. "Rerouting cleanly" means the requests in flight on the dead replica are retried or shifted onto a healthy one so the user still gets an answer; "dropping requests" means those users simply get an error.

## Why This Matters

In production a GPU will die mid-request and the user's [KV cache](/shared/glossary/#kv-cache) is lost. Rehearsing the failure tells you whether your routing layer reroutes cleanly or drops requests — long before it happens for real at 3 a.m.
