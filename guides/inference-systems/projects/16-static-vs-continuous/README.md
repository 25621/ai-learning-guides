# Static vs. Continuous

---

> Static batching makes everyone wait for the slowest request; continuous batching lets each one leave the moment it is done.

---

## Key Insight

This project builds two batching strategies around a toy [decode](/shared/glossary/#decode) loop — *static* batching (gather a fixed group, run them together, and they all finish together) and [continuous batching](/shared/glossary/#continuous-batching) (add and drop requests every step) — then load-tests both under requests of varying lengths and arrival times.

## Why This Matters

Static batching wastes the GPU whenever requests differ in length, because short ones sit idle waiting for long ones. Continuous batching keeps the GPU full and is the single biggest [throughput](/shared/glossary/#throughput) win in modern serving — building both yourself makes the gap impossible to forget.
