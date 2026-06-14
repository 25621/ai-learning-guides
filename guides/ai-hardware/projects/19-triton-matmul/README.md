# Triton Matmul

## Key Insight

Implementing a tile-based [matrix multiplication](/shared/glossary/#matmul) in [Triton](/shared/glossary/#triton) shows how the language automatically handles memory layouts and [shared memory](/shared/glossary/#shared-memory) scheduling while allowing the user to control [tiling](/shared/glossary/#tiling) logic. By structuring the algorithm as a sequence of block-level dot products, we optimize register reuse and data transfer. Sweeping over different block sizes and configuring autotuning yields execution performance that approaches optimized [cuBLAS](/shared/glossary/#cublas) throughput.
