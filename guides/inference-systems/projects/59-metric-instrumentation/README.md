# Metric Instrumentation

---

> "You can't fix what you can't see — give the server a set of gauges before you touch anything else."

---

## Key Insight

Metric instrumentation means adding code to a serving engine (like [vLLM](/shared/glossary/#vllm)) so it continuously reports numbers about its own behavior — [latency](/shared/glossary/#latency), [throughput](/shared/glossary/#throughput), error rate, GPU usage — which tools such as Prometheus and Grafana then collect and graph. Watching [percentiles](/shared/glossary/#percentile) like p99, not just averages, is what reveals the slow requests real users actually feel.

## Why This Matters

Without live metrics a problem only surfaces when a customer complains. Wiring up the right signals is the foundation of [observability](/shared/glossary/#observability) — the basis on which every dashboard, alert, and capacity decision is built.
