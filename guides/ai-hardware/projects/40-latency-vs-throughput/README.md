# Latency vs Throughput

## Key Insight

Large language model serving exhibits a fundamental trade-off between user-perceived response times and server processing efficiency. While a small batch size minimizes [TTFT (Time To First Token)](/shared/glossary/#ttft) and [ITL / TPOT](/shared/glossary/#itl--tpot), larger batches increase the system's [throughput](/shared/glossary/#throughput) at the cost of higher individual latency. Plotting this relationship helps identify the optimal operational point, or "knee," where hardware utilization is maximized without violating latency SLA budgets.
