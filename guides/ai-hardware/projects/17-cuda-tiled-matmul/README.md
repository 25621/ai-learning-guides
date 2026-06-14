# CUDA Tiled Matmul

## Key Insight

Implementing a [tiled](/shared/glossary/#tiling) [matrix multiplication](/shared/glossary/#matmul) in [CUDA](/shared/glossary/#cuda) shows how staging data in fast [shared memory](/shared/glossary/#shared-memory) minimizes slow global memory transactions. By partitioning matrices into blocks that fit within the Streaming Multiprocessor ([SM](/shared/glossary/#sm))'s on-chip storage, we reuse loaded data across multiple threads to increase arithmetic intensity. Tuning tile dimensions is key to achieving a high percentage of optimized library performance like [cuBLAS](/shared/glossary/#cublas).
