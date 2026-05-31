# Failure-Mode Drill

---

> Pull the plug on a replica under load and see whether users even notice.

---

## Key Insight

This project kills one replica in the middle of a load test and checks that the system fails over gracefully — rerouting traffic to the healthy replicas — while measuring the user-visible impact.

## Why This Matters

In production a GPU will die mid-request and the user's [KV cache](/shared/glossary/#kv-cache) is lost. Rehearsing the failure tells you whether your routing layer reroutes cleanly or drops requests — long before it happens for real at 3 a.m.
