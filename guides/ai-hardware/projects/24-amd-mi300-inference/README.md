# AMD MI300 Inference

## Key Insight

Deploying large language models on AMD's [MI300X](/shared/glossary/#amd-instinct-mi300x) accelerator demonstrates the capability of alternative hardware to deliver high [throughput](/shared/glossary/#throughput) and capacity outside the NVIDIA ecosystem. By leveraging the open-source [ROCm](/shared/glossary/#rocm) stack and runtime engines like [vLLM](/shared/glossary/#vllm), developers can target AMD's silicon architecture natively or port existing configurations via [HIP](/shared/glossary/#hip). This provides a viable pathway to mitigate GPU supply constraints while maintaining high performance for memory-bound serving workloads.
