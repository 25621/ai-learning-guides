# FSDP Scaling

## Key Insight

[FSDP (Fully Sharded Data Parallel)](/shared/glossary/#fsdp) shards a model's parameters, [gradients](/shared/glossary/#gradients), and [optimizer state](/shared/glossary/#optimizer-state) across all participating [GPUs](/shared/glossary/#gpu) in the distributed group. To run a layer's computation, FSDP uses an [AllGather](/shared/glossary/#allgather) collective operation to temporarily reconstruct the sharded weights, executes the forward or backward pass, and then immediately discards the full weights. By replacing DDP's large [AllReduce](/shared/glossary/#allreduce) with interleaved [AllGather](/shared/glossary/#allgather) and [ReduceScatter](/shared/glossary/#reducescatter) collectives, FSDP reduces per-GPU memory usage from $O(1)$ to $O(1/N)$ where $N$ is the [world size](/shared/glossary/#rank), allowing developers to scale training to massive model architectures.
