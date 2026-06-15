# Continuous Batching Demo

## Key Insight

Static batching is highly inefficient for serving language models due to the wide variability in prompt and generation lengths. Implementing [continuous batching](/shared/glossary/#continuous-batching) allows the serving engine to dynamically insert new requests and extract completed ones at the granularity of individual token steps. This project demonstrates how this scheduling strategy maximizes GPU utilization and increases overall throughput compared to waiting for entire batches to complete.
