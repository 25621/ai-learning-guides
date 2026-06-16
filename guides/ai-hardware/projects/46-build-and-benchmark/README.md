# Build and Benchmark

## Key Insight

Assembling a multi-[GPU](/shared/glossary/#gpu) workstation and running [NCCL](/shared/glossary/#nccl) benchmarks reveals the gap between a component's theoretical bandwidth and its real-world performance. An [AllReduce](/shared/glossary/#allreduce) throughput test with `nccl-tests` measures how quickly the GPUs can synchronize [gradients](/shared/glossary/#gradients) across the [PCIe](/shared/glossary/#pcie) bus — the step that dominates wall-clock time during [distributed training](/shared/glossary/#data-parallelism). Low numbers point to concrete problems: a GPU seated in an ×8 slot instead of ×16, a misconfigured BIOS setting, or a faulty riser cable. Catching these issues before starting a multi-day training run saves both time and electricity.
