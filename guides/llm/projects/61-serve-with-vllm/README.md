# Serve with vLLM

---

> Go from a notebook script to an HTTP server in one engine.

---

## Key Insight

This project stands up a [vLLM](/shared/glossary/#vllm) server in front of an open model and load-tests it with many simultaneous requests, so you can directly observe [continuous batching](/shared/glossary/#continuous-batching) and [PagedAttention](/shared/glossary/#pagedattention) at work on a real model.

## Why This Matters

A serving engine like vLLM turns the slow per-user, one-token-at-a-time decode into hundreds of tokens per second across many concurrent users by dynamically merging their requests — the engineering gap between a demo script and a real production system.
