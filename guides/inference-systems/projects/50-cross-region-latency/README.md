# Cross-Region Latency

---

> The closest datacenter wins the first token; physics does the rest.

---

## Key Insight

This project deploys the same model in two geographic regions and measures the [time to first token](/shared/glossary/#ttft) difference when each request is routed to the nearest region versus a far one.

## Why This Matters

Network round-trips across continents add tens to hundreds of milliseconds before any compute even starts, so routing each user to their closest region is often the cheapest [latency](/shared/glossary/#latency) win available — with no change to the model at all.
