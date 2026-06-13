# GPU vs CPU Bake-off

## Key Insight

Comparing [matrix multiplications](/shared/glossary/#matmul) on a [CPU](/shared/glossary/#cpu) using [NumPy](/shared/glossary/#numpy) against a [GPU](/shared/glossary/#gpu) using [PyTorch](/shared/glossary/#pytorch) highlights the trade-off between latency and throughput. The GPU leverages thousands of simpler cores executing under a [SIMT](/shared/glossary/#simt) model to achieve massive parallel speedups, whereas the CPU relies on a few fast cores optimized for quick single-threaded tasks. This head-to-head comparison shows exactly when the overhead of copying data to the GPU is justified by the scale of the workload.
