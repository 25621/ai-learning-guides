# Workload Sensitivity

---

> Speculation flies on predictable text and stalls on surprising text.

---

## Key Insight

This project measures speculative-decoding speedup across very different workloads — chat, code completion, summarization, and [constrained](/shared/glossary/#constrained-generation) JSON output — and explains why the [acceptance rate](/shared/glossary/#acceptance-rate), and therefore the speedup, varies so much between them.

## Why This Matters

How predictable the next token is decides how often the draft guesses right, so the same system can see 3× on copy-heavy code yet barely 1.3× on open-ended chat. Knowing your workload's acceptance rate before you promise a latency number keeps you from over-claiming.
