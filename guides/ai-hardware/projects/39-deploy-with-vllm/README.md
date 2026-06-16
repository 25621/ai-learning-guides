# Deploy with vLLM

## Key Insight

Serving large language models requires optimizing memory allocation for dynamic sequence lengths. By deploying a model using [vLLM](/shared/glossary/#vllm), this project demonstrates how [PagedAttention](/shared/glossary/#pagedattention) prevents physical memory fragmentation of the [KV cache](/shared/glossary/#kv-cache). Measuring throughput across various batch sizes reveals how amortizing weight loads from [HBM (High-Bandwidth Memory)](/shared/glossary/#hbm) over concurrent requests increases overall generation efficiency.
