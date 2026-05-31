# Priority Queue

---

> When the server is busy, the requests that matter most should be able to jump the line.

---

## Key Insight

This project adds priority classes to [vLLM](/shared/glossary/#vllm)'s [scheduler](/shared/glossary/#scheduler) so high-priority requests are picked before others, then checks that their [TTFT](/shared/glossary/#ttft) stays low even while the server is under heavy load.

## Why This Matters

Real services mix urgent and background work — paid versus free, interactive versus batch. A priority queue lets the important requests move ahead so they meet their [latency](/shared/glossary/#latency) targets, while cheaper work waits its turn instead of crowding them out.
