# Multi-Node Setup

## Key Insight

Scaling deep learning training beyond a single server requires a [multi-node](/shared/glossary/#node-distributed) configuration, where virtual or physical machines are connected via a high-speed network. In this environment, the communication bottleneck shifts from fast intra-node interconnects like [NVLink](/shared/glossary/#nvlink) to slower inter-node connections like [InfiniBand (IB)](/shared/glossary/#infiniband-ib) or [RoCE](/shared/glossary/#roce). To prevent the network from stalling the training loop, distributed frameworks rely on optimized [network topologies](/shared/glossary/#network-topology) and collective communication libraries like [NCCL](/shared/glossary/#nccl) to pipeline data transfer and maximize throughput.
