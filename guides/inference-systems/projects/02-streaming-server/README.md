# Streaming Server

---

> A user does not feel "tokens per second" — they feel the wait for the first word and the gaps between the rest.

---

## Key Insight

This project wraps the manual decode loop in a FastAPI endpoint that [streams](/shared/glossary/#streaming) tokens to the client as they are generated, then load-tests it to measure two distinct numbers: [TTFT](/shared/glossary/#ttft) (how long until the first token arrives) and [ITL](/shared/glossary/#itl--tpot) (how long between each token after that). The two latencies feel different to a user and respond to different optimizations.

## Why This Matters

A serving system that quotes only an average latency is hiding the user experience. Long answers feel snappy if [TTFT](/shared/glossary/#ttft) is low; short answers feel slow if [ITL](/shared/glossary/#itl--tpot) is high — and you cannot fix what you do not measure. Splitting the two numbers from day one is the foundation of every production [SLO](/shared/glossary/#slo) in this guide.
