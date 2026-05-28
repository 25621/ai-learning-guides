# Prefix-Cache Study

---

> Stop redoing the same long system prompt for every request.

---

## Key Insight

This project runs the same workload twice on a [vLLM](/shared/glossary/#vllm) server — once with the [prefix cache](/shared/glossary/#prefix-cache) turned on and once with it turned off — then compares the [tail latency](/shared/glossary/#tail-latency) and the [TTFT](/shared/glossary/#ttft) for requests that share a long system message.

## Why This Matters

Real traffic is mostly a long shared system prompt followed by a short user turn, so caching the keys and values for that prefix means the boilerplate is processed only once across many users — a quiet but huge throughput win whenever the prompts your users send have a fixed beginning.
