# Occupancy Study

## Key Insight

Measuring Streaming Multiprocessor ([SM](/shared/glossary/#sm)) [occupancy](/shared/glossary/#occupancy) shows the ratio of active [warps](/shared/glossary/#warp) to the maximum supported warps on the hardware. Sweeping block size in a custom [CUDA](/shared/glossary/#cuda) kernel demonstrates how resource limits, such as register allocation and shared memory size, constrain parallelism and affect the GPU's ability to hide memory latency.
