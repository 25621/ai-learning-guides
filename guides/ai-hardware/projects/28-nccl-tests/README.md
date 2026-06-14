# NCCL Tests

## Key Insight

Running NVIDIA's official [NCCL (NVIDIA Collective Communications Library)](/shared/glossary/#nccl) benchmarks allows developers to isolate and measure raw [AllReduce](/shared/glossary/#allreduce) and [AllGather](/shared/glossary/#allgather) bandwidth across multiple [GPUs](/shared/glossary/#gpu). By bypassing training frameworks and model compute overhead, these tests establish a clean hardware baseline for communication latency and throughput over [NVLink](/shared/glossary/#nvlink) or [InfiniBand (IB)](/shared/glossary/#infiniband-ib) networks. This isolation is crucial for diagnosing network hardware issues, driver bottlenecks, or topology misconfigurations before launching expensive, large-scale training jobs.
