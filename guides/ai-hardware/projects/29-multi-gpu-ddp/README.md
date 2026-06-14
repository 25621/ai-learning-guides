# Multi-GPU DDP

## Key Insight

Training models with [DDP (Distributed Data Parallel)](/shared/glossary/#ddp) replicates the model across multiple [GPUs](/shared/glossary/#gpu), where each GPU processes a different slice of the training [batch](/shared/glossary/#batch) in parallel. During the [backward pass](/shared/glossary/#backward-pass), the GPUs perform an [AllReduce](/shared/glossary/#allreduce) collective operation to synchronize and sum their [gradients](/shared/glossary/#gradients) before updating weights. Profiling the execution shows that while computation scales linearly with the number of GPUs, communication overhead from gradient synchronization can bottleneck training if network bandwidth is insufficient.
