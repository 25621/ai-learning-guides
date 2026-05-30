# FP4 (Blackwell) Deployment

---

> Four bits per weight, accelerated in hardware — if you can keep the quality.

---

## Key Insight

On [Blackwell](/shared/glossary/#blackwell) GPUs that support it in hardware, this project benchmarks [FP4](/shared/glossary/#fp4) [weights](/shared/glossary/#weights) against [FP8](/shared/glossary/#fp8), reporting the [throughput](/shared/glossary/#throughput) gain, any quality loss, and the operational gotchas of running such a new format.

## Why This Matters

FP4 halves weight size again versus FP8, promising more speed and more concurrency — but 4-bit floating point sits close to the edge of usable precision, so you must measure carefully before trusting it in production.
