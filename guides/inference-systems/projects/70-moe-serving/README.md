# MoE Serving

---

> An [MoE](/shared/glossary/#moe) is enormous on paper but cheap per token — until one [expert](/shared/glossary/#expert) gets all the traffic and the others sit idle.

---

## Key Insight

This project stands up a [Mixture-of-Experts](/shared/glossary/#moe) model (such as Mixtral or a DeepSeek MoE) and measures expert *imbalance* under a real workload — how unevenly the router spreads tokens across the [experts](/shared/glossary/#expert). When experts are split across GPUs with [expert parallelism](/shared/glossary/#expert-parallelism-ep), a lopsided distribution means some GPUs are overworked while others wait, capping throughput.

## Why This Matters

MoE models give you huge capacity at a fixed compute cost per token, but only if the experts stay evenly busy. Imbalance is the dominant serving headache: it turns the all-to-all token routing on every step into a bottleneck and wastes the very hardware you added experts to use. Measuring it on your own traffic is the first step to tuning capacity factors and placement.
