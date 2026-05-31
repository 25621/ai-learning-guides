# Speculation + Batching

---

> Every request in the batch accepts a different number of guesses — the engine has to handle the ragged edge.

---

## Key Insight

This project combines [speculative decoding](/shared/glossary/#speculative-decoding) with [continuous batching](/shared/glossary/#continuous-batching): each request in the [batch](/shared/glossary/#batch) accepts a different number of draft tokens per step, so the engine must handle this "ragged" result and re-pack the batch without stalling on the unluckiest request.

## Why This Matters

Speculation and batching are the two biggest decode speedups, yet they pull against each other — a naive batch runs only as fast as its least-lucky request. Making them cooperate is what lets a production server win higher [throughput](/shared/glossary/#throughput) and lower latency at the same time.
